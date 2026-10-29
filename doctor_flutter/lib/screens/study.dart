import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api.dart';
import '../widgets/common.dart';

/// One CT study: AI summary, slice viewer (windows, risk bar, heatmap, finding circles),
/// findings with review / nodule features / cancer risks, impression, release and report.
class StudyScreen extends StatefulWidget {
  final int studyId;
  const StudyScreen({super.key, required this.studyId});

  @override
  State<StudyScreen> createState() => _StudyScreenState();
}

class _StudyScreenState extends State<StudyScreen> {
  final api = Api.instance;
  Map<String, dynamic>? _s;
  String? _error;
  Timer? _poll;

  // viewer state
  int _z = 0;
  String _win = 'lung';
  bool _cam = false;
  int? _sel;
  Uint8List? _img;
  final Map<String, Uint8List> _sliceCache = {};
  final Map<String, Uint8List?> _camCache = {};
  final Set<String> _loading = {};
  final FocusNode _viewerFocus = FocusNode();

  final _impression = TextEditingController();
  final Map<int, TextEditingController> _notes = {};
  String _impMsg = '';

  int get sid => widget.studyId;
  Map<String, dynamic>? get _rep => _s?['report'] as Map<String, dynamic>?;
  Map<String, dynamic> get _geo => _rep!['geometry'] as Map<String, dynamic>;
  List<Map<String, dynamic>> get _findings =>
      ((_rep?['findings'] as List<dynamic>?) ?? []).map((e) => e as Map<String, dynamic>).toList();
  Map<String, dynamic> get _reviews => (_s?['reviews'] as Map<String, dynamic>?) ?? {};
  Map<String, dynamic>? _review(int fid) => _reviews['$fid'] as Map<String, dynamic>?;
  bool _rejected(int fid) => _review(fid)?['decision'] == 'reject';
  bool get _busy => ['queued', 'processing', 'uploading'].contains(_s?['status']);

  @override
  void initState() {
    super.initState();
    _refresh(first: true);
    _poll = Timer.periodic(const Duration(milliseconds: 2500), (_) {
      if (_busy) _refresh();
    });
  }

  @override
  void dispose() {
    _poll?.cancel();
    _viewerFocus.dispose();
    super.dispose();
  }

  Future<void> _refresh({bool first = false}) async {
    try {
      final s = await api.send('GET', '/api/studies/$sid') as Map<String, dynamic>;
      if (!mounted) return;
      final becameDone = s['status'] == 'done' && (first || _s?['status'] != 'done');
      setState(() {
        _s = s;
        _error = null;
      });
      if (becameDone) _initViewer();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  void _initViewer() {
    _impression.text = '${_s?['impression'] ?? ''}';
    _impMsg = _s?['reviewed_at'] != null ? 'saved by ${_s?['reviewed_by'] ?? '?'} at ${fmtDate(_s?['reviewed_at'])}' : '';
    for (final f in _findings) {
      final id = asInt(f['id']);
      _notes.putIfAbsent(id, () => TextEditingController()).text = '${_review(id)?['note'] ?? ''}';
    }
    final first = _findings.isNotEmpty ? _findings.first : null;
    _sliceCache.clear();
    _camCache.clear();
    _sel = first == null ? null : asInt(first['id']);
    _setSlice(first == null ? asInt(_geo['slices']) ~/ 2 : asInt((first['voxel_zyx'] as List)[0]));
  }

  // ------------------------------------------------------------------ slices and heatmaps
  String _key(int z) => '$_win:$z';

  Future<Uint8List?> _fetchSlice(int z) async {
    final k = _key(z);
    if (_sliceCache.containsKey(k)) return _sliceCache[k];
    if (_loading.contains(k)) return null;
    _loading.add(k);
    try {
      final b = await api.bytes('/api/studies/$sid/slice?z=$z&window=$_win');
      if (b != null) {
        _sliceCache[k] = b;
        if (_sliceCache.length > 150) _sliceCache.remove(_sliceCache.keys.first);
      }
      return b;
    } catch (_) {
      return null;
    } finally {
      _loading.remove(k);
    }
  }

  void _setSlice(int z) {
    if (_rep == null) return;
    final n = asInt(_geo['slices']);
    final zz = math.min(math.max(z, 0), n - 1);
    setState(() {
      _z = zz;
      final cached = _sliceCache[_key(zz)];
      if (cached != null) _img = cached;
    });
    if (!_sliceCache.containsKey(_key(zz))) {
      _fetchSlice(zz).then((b) {
        if (mounted && b != null && _z == zz) setState(() => _img = b);
      });
    }
    for (final d in const [1, 2, 3, -1, -2]) {
      final p = zz + d;
      if (p >= 0 && p < n) _fetchSlice(p);
    }
    if (_cam) _fetchCams(zz);
  }

  void _fetchCams(int z) {
    final sz = (asDouble((_geo['spacing_xyz_mm'] as List)[2]) ?? 1.0).abs();
    for (final f in _findings) {
      if (f['cam'] == null) continue;
      final fz = asInt((f['voxel_zyx'] as List)[0]);
      if ((fz - z).abs() * sz > 16) continue;
      final k = '${f['id']}:$z';
      if (_camCache.containsKey(k) || _loading.contains('cam$k')) continue;
      _loading.add('cam$k');
      api.bytes('/api/studies/$sid/cam/${f['id']}?z=$z').then((b) {
        _camCache[k] = b;
        if (_camCache.length > 300) _camCache.remove(_camCache.keys.first);
        if (mounted && _z == z) setState(() {});
      }).catchError((_) {
        _camCache[k] = null;
      }).whenComplete(() => _loading.remove('cam$k'));
    }
  }

  void _setWindow(String w) {
    setState(() => _win = w);
    _setSlice(_z);
  }

  /// Slice-level risk: each slice gets the highest nodule probability of the (non-rejected) findings
  /// passing through it; green < yellow threshold <= yellow < red threshold <= red (proposal: 30% / 70%).
  List<double> _sliceRisk() {
    final n = asInt(_geo['slices']);
    final sz = (asDouble((_geo['spacing_xyz_mm'] as List)[2]) ?? 1.0).abs();
    final risk = List<double>.filled(n, 0);
    for (final f in _findings) {
      if (_rejected(asInt(f['id']))) continue;
      final rz = math.max(1, (math.max(3.0, asDouble(f['diameter_mm'])! / 2) / sz).ceil());
      final fz = asInt((f['voxel_zyx'] as List)[0]);
      final p = asDouble(f['probability'])!;
      for (var z = fz - rz; z <= fz + rz; z++) {
        if (z >= 0 && z < n && p > risk[z]) risk[z] = p;
      }
    }
    return risk;
  }

  // ------------------------------------------------------------------ actions
  Future<void> _decide(int fid, String decision) async {
    final note = _notes[fid]?.text.trim();
    try {
      await api.send('PUT', '/api/studies/$sid/findings/$fid/review',
          body: {'decision': decision, 'note': (note == null || note.isEmpty) ? null : note});
      setState(() {
        final rv = Map<String, dynamic>.from(_reviews);
        rv['$fid'] = {'decision': decision, 'note': note};
        _s!['reviews'] = rv;
      });
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _setAttrs(int fid, Map<String, dynamic> attrs) async {
    try {
      final s = await api.send('PUT', '/api/studies/$sid/findings/$fid/attrs', body: attrs) as Map<String, dynamic>;
      setState(() => _s!['report'] = s['report']);
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _saveImpression() async {
    try {
      await api.send('PUT', '/api/studies/$sid/impression', body: {'impression': _impression.text});
      _s!['impression'] = _impression.text;
      setState(() => _impMsg = 'saved ✓');
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _release() async {
    final released = _s!['released'] == 1;
    try {
      await api.send('POST', '/api/studies/$sid/release?released=${!released}');
      await _refresh();
      if (mounted && !released) {
        toast(context, 'Released. The patient can now see this study in the patient app '
            '(access code ${(_s!['patient'] as Map)['access_code']}).');
      }
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _reanalyze() async {
    final ok = await confirmDialog(context, 'Re-analyze', 'Run the AI on this scan again with the current settings?');
    if (!ok) return;
    try {
      await api.send('POST', '/api/studies/$sid/reanalyze');
      await _refresh(first: true);
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _printReport() async {
    final url = Uri.parse(api.linkWithToken('/report/$sid'));
    if (!await launchUrl(url, mode: LaunchMode.externalApplication, webOnlyWindowName: '_blank')) {
      if (mounted) toast(context, 'Could not open the report: $url', error: true);
    }
  }

  // ------------------------------------------------------------------ UI
  @override
  Widget build(BuildContext context) {
    final s = _s;
    final p = s?['patient'] as Map<String, dynamic>?;
    return Scaffold(
      appBar: AppBar(
        title: Text(s == null ? 'CT study #$sid' : '${p?['name'] ?? ''} · CT study #$sid'),
        actions: s == null
            ? null
            : [
                TextButton.icon(onPressed: _busy ? null : _reanalyze, icon: const Icon(Icons.replay), label: const Text('Re-analyze')),
                TextButton.icon(
                    onPressed: s['status'] == 'done' ? _printReport : null,
                    icon: const Icon(Icons.print_outlined),
                    label: const Text('Print report')),
                const SizedBox(width: 6),
                FilledButton.icon(
                  onPressed: s['status'] == 'done' ? _release : null,
                  icon: Icon(s['released'] == 1 ? Icons.undo : Icons.send_outlined),
                  label: Text(s['released'] == 1 ? 'Released ✓ (undo)' : 'Release to patient'),
                ),
                const SizedBox(width: 12),
              ],
      ),
      body: s == null
          ? Center(child: _error != null ? Text(_error!, style: const TextStyle(color: kRed)) : const CircularProgressIndicator())
          : _body(s),
    );
  }

  Widget _body(Map<String, dynamic> s) {
    if (_busy) {
      final processing = s['status'] == 'processing';
      return Center(
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(28),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(processing ? 'Analysing the scan…' : 'Waiting in queue…', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 6),
              Text(processing
                  ? 'Finding candidates in the lungs and scoring them with the AI model (about a minute).'
                  : '${s['queue_position'] ?? 0} study(ies) ahead.'),
            ]),
          ),
        ),
      );
    }
    if (s['status'] == 'error') {
      return Center(
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              const Text('Analysis failed', style: TextStyle(color: kRed, fontWeight: FontWeight.bold, fontSize: 18)),
              const SizedBox(height: 8),
              Text('${s['error'] ?? ''}'),
              const SizedBox(height: 12),
              FilledButton(
                  onPressed: () async {
                    await api.send('POST', '/api/studies/$sid/reanalyze');
                    _refresh(first: true);
                  },
                  child: const Text('Retry')),
            ]),
          ),
        ),
      );
    }
    if (_rep == null) return const Center(child: Text('No AI report.'));
    return LayoutBuilder(builder: (context, c) {
      final wide = c.maxWidth >= 1100;
      final viewer = _viewerCard();
      final side = Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        _findingsCard(),
        const SizedBox(height: 16),
        _impressionCard(),
      ]);
      return SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('${fmtDate(s['created'])} · ${s['source'] ?? ''}',
              style: Theme.of(context).textTheme.bodySmall, maxLines: 1, overflow: TextOverflow.ellipsis),
          const SizedBox(height: 10),
          _tiles(),
          const SizedBox(height: 12),
          _statement(),
          const SizedBox(height: 16),
          if (wide)
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Expanded(flex: 6, child: viewer),
              const SizedBox(width: 16),
              Expanded(flex: 5, child: side),
            ])
          else ...[
            viewer,
            const SizedBox(height: 16),
            side,
          ],
        ]),
      );
    });
  }

  Widget _tiles() {
    final sm = _rep!['summary'] as Map<String, dynamic>;
    final timing = sm['timing'] as Map<String, dynamic>?;
    final flagged = _findings.where((f) => f['needs_review'] == true).length;
    final largest = asDouble(sm['largest_nodule_mm']) ?? 0;
    Widget tile(String k, String v, {String? tip}) {
      final w = Container(
        width: 190,
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
            color: Colors.white, border: Border.all(color: const Color(0xFFE0E0E0)), borderRadius: BorderRadius.circular(8)),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(k, style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: 4),
          Text(v, style: Theme.of(context).textTheme.headlineSmall),
        ]),
      );
      return tip == null ? w : Tooltip(message: tip, child: w);
    }

    return Wrap(spacing: 12, runSpacing: 12, children: [
      tile('Suspected nodules', '${sm['n_nodules_detected']}'),
      tile('Largest nodule', largest > 0 ? '${largest.toStringAsFixed(1)} mm' : '–'),
      tile('Flagged for review', '$flagged'),
      tile('Candidates screened', '${sm['n_candidates_screened']}'),
      tile('Analysis time', '${sm['runtime_s']} s',
          tip: timing?.entries.map((e) => '${e.key.replaceAll(RegExp(r'_s$'), '').replaceAll('_', ' ')}: ${e.value} s').join('\n')),
    ]);
  }

  Widget _statement() {
    final sm = _rep!['summary'] as Map<String, dynamic>;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(left: BorderSide(color: kBrandBlue, width: 3)),
      ),
      child: Text('${sm['statement']} ${sm['disclaimer']}'),
    );
  }

  // ---------------------------------------------------------------- viewer
  Widget _viewerCard() {
    final g = _geo;
    final n = asInt(g['slices']);
    final rows = asInt(g['rows']);
    final cols = asInt(g['cols']);
    final st = (_s!['settings'] as Map<String, dynamic>?) ?? {};
    final yellow = asDouble(st['slice_yellow']) ?? 0.3;
    final red = asDouble(st['slice_red']) ?? 0.7;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: const Color(0xFF111111), borderRadius: BorderRadius.circular(10)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Row(children: [
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'lung', label: Text('Lung')),
              ButtonSegment(value: 'soft', label: Text('Soft tissue')),
              ButtonSegment(value: 'bone', label: Text('Bone')),
            ],
            selected: {_win},
            showSelectedIcon: false,
            onSelectionChanged: (v) => _setWindow(v.first),
            style: SegmentedButton.styleFrom(
              foregroundColor: Colors.white70,
              selectedForegroundColor: Colors.white,
              selectedBackgroundColor: kBrandBlue,
            ),
          ),
          const Spacer(),
          Checkbox(
            value: _cam,
            onChanged: (v) {
              setState(() => _cam = v ?? false);
              if (_cam) _fetchCams(_z);
            },
            side: const BorderSide(color: Colors.white70),
          ),
          const Text('AI heatmap', style: TextStyle(color: Colors.white70)),
          const SizedBox(width: 12),
          Text('slice ${_z + 1} / $n', style: const TextStyle(color: Color(0xFFF2C14E))),
        ]),
        const SizedBox(height: 10),
        Focus(
          focusNode: _viewerFocus,
          autofocus: true,
          onKeyEvent: (node, e) {
            if (e is KeyDownEvent || e is KeyRepeatEvent) {
              if (e.logicalKey == LogicalKeyboardKey.arrowUp) {
                _setSlice(_z - 1);
                return KeyEventResult.handled;
              }
              if (e.logicalKey == LogicalKeyboardKey.arrowDown) {
                _setSlice(_z + 1);
                return KeyEventResult.handled;
              }
            }
            return KeyEventResult.ignored;
          },
          child: Listener(
            onPointerDown: (_) => _viewerFocus.requestFocus(),
            onPointerSignal: (e) {
              if (e is PointerScrollEvent) {
                GestureBinding.instance.pointerSignalResolver.register(e, (ev) {
                  final se = ev as PointerScrollEvent;
                  _setSlice(_z + (se.scrollDelta.dy > 0 ? 1 : -1));
                });
              }
            },
            child: GestureDetector(
              // drag up / down on a touch screen moves through the slices
              onVerticalDragUpdate: (d) {
                if (d.delta.dy.abs() > 2) _setSlice(_z + (d.delta.dy > 0 ? 1 : -1));
              },
              // fit the scan into the window height, so image, slider and risk bar are visible together
              child: Center(
                child: ConstrainedBox(
                  constraints: BoxConstraints(maxHeight: math.max(300.0, MediaQuery.of(context).size.height - 260)),
                  child: AspectRatio(
                    aspectRatio: cols / rows,
                    child: LayoutBuilder(builder: (context, c) => _imageStack(c.maxWidth / cols)),
                  ),
                ),
              ),
            ),
          ),
        ),
        Slider(
          value: _z.toDouble(),
          min: 0,
          max: math.max(1, n - 1).toDouble(),
          onChanged: (v) => _setSlice(v.round()),
        ),
        LayoutBuilder(builder: (context, c) {
          return GestureDetector(
            onTapDown: (d) => _setSlice((d.localPosition.dx / c.maxWidth * n).floor()),
            onHorizontalDragUpdate: (d) => _setSlice((d.localPosition.dx / c.maxWidth * n).floor()),
            child: CustomPaint(
              size: Size(c.maxWidth, 18),
              painter: _RiskBarPainter(_sliceRisk(), _z, yellow, red),
            ),
          );
        }),
        const SizedBox(height: 6),
        Wrap(spacing: 12, crossAxisAlignment: WrapCrossAlignment.center, children: [
          _legend(kGreen, 'clear'),
          _legend(kYellow, 'uncertain'),
          _legend(kRed, 'suspicious'),
          Text('(yellow ≥ ${(yellow * 100).round()}%, red ≥ ${(red * 100).round()}% nodule probability)',
              style: const TextStyle(color: Colors.white54, fontSize: 12)),
        ]),
        const SizedBox(height: 4),
        const Text('Scroll the mouse wheel over the image, drag on a touch screen, or use ↑ ↓. Circles mark AI findings.',
            style: TextStyle(color: Colors.white54, fontSize: 12)),
      ]),
    );
  }

  Widget _legend(Color c, String t) => Row(mainAxisSize: MainAxisSize.min, children: [
        Container(width: 10, height: 10, color: c),
        const SizedBox(width: 4),
        Text(t, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ]);

  Widget _imageStack(double scale) {
    final g = _geo;
    final sp = (g['spacing_xyz_mm'] as List).map((e) => (asDouble(e) ?? 1.0).abs()).toList();
    final sx = sp[0], sy = sp[1], sz = sp[2];
    final cams = <Widget>[];
    if (_cam) {
      for (final f in _findings) {
        if (f['cam'] == null) continue;
        final v = (f['voxel_zyx'] as List).map((e) => asInt(e)).toList();
        if ((v[0] - _z).abs() * sz > 16) continue;
        final b = _camCache['${f['id']}:$_z'];
        if (b == null) continue;
        cams.add(Positioned(
          left: (v[2] - 16 / sx) * scale,
          top: (v[1] - 16 / sy) * scale,
          width: 32 / sx * scale,
          height: 32 / sy * scale,
          child: IgnorePointer(child: Image.memory(b, fit: BoxFit.fill, gaplessPlayback: true)),
        ));
      }
    }
    return ClipRect(
      child: Stack(children: [
        Positioned.fill(child: Container(color: Colors.black)),
        if (_img != null) Positioned.fill(child: Image.memory(_img!, fit: BoxFit.fill, gaplessPlayback: true)),
        if (_img == null) const Center(child: CircularProgressIndicator()),
        ...cams,
        Positioned.fill(
          child: IgnorePointer(
            child: CustomPaint(
              painter: _FindingsPainter(
                findings: _findings,
                z: _z,
                sx: sx,
                sz: sz,
                scale: scale,
                selected: _sel,
                rejected: {for (final f in _findings) if (_rejected(asInt(f['id']))) asInt(f['id'])},
              ),
            ),
          ),
        ),
      ]),
    );
  }

  // ---------------------------------------------------------------- findings
  Widget _findingsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const SectionTitle('Findings (tap to jump to the slice)'),
          if (_findings.isEmpty) const Text('No suspected nodules above the threshold. Please still review the scan.'),
          for (final f in _findings) ...[_findingTile(f), const SizedBox(height: 10)],
        ]),
      ),
    );
  }

  Widget _findingTile(Map<String, dynamic> f) {
    final id = asInt(f['id']);
    final sel = _sel == id;
    final rv = _review(id);
    final cat = '${f['lung_rads_size_category']}';
    final attrs = (f['attrs'] as Map<String, dynamic>?) ?? {'nodule_type': 'solid', 'upper_lobe': false, 'spiculated': false};
    final prob = asDouble(f['probability'])!;
    final v = (f['voxel_zyx'] as List).map((e) => asInt(e)).toList();
    final small = Theme.of(context).textTheme.bodySmall;
    Widget decide(String d, String label, IconData icon) {
      final on = rv?['decision'] == d;
      return on
          ? FilledButton.tonalIcon(onPressed: () => _decide(id, d), icon: Icon(icon, size: 16), label: Text(label))
          : OutlinedButton.icon(onPressed: () => _decide(id, d), icon: Icon(icon, size: 16), label: Text(label));
    }

    return InkWell(
      borderRadius: BorderRadius.circular(8),
      onTap: () {
        setState(() => _sel = id);
        _setSlice(v[0]);
      },
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          border: Border.all(color: sel ? kBrandBlue : const Color(0xFFDDDDDD), width: sel ? 2 : 1),
          borderRadius: BorderRadius.circular(8),
          color: _rejected(id) ? const Color(0xFFF4F4F4) : Colors.white,
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Row(children: [
            Text('Finding $id', style: const TextStyle(fontWeight: FontWeight.bold)),
            if (f['above_threshold'] != true) Text('  (below threshold)', style: small),
            const Spacer(),
            Flexible(child: Pill('Lung-RADS $cat', radsColor(cat))),
          ]),
          const SizedBox(height: 8),
          Wrap(spacing: 16, runSpacing: 6, crossAxisAlignment: WrapCrossAlignment.center, children: [
            Row(mainAxisSize: MainAxisSize.min, children: [
              Text('Nodule probability ${pct(prob)} '),
              SizedBox(width: 60, child: LinearProgressIndicator(value: prob, minHeight: 5)),
            ]),
            Text('Size ${asDouble(f['diameter_mm'])!.toStringAsFixed(1)} mm'),
            Text('Slice ${v[0]}'),
            Text('Uncertainty ±${pct(f['uncertainty'])}'),
          ]),
          const SizedBox(height: 6),
          Wrap(spacing: 16, runSpacing: 6, children: [
            if (f['cancer_risk'] != null)
              Tooltip(
                message: 'Image-based AI estimate (model trained on LUNA25 / NLST screening). Not a diagnosis.',
                child: Text.rich(TextSpan(children: [
                  const TextSpan(text: 'AI cancer risk '),
                  TextSpan(
                      text: '${pct(f['cancer_risk'])} · ${f['cancer_risk_band']}',
                      style: TextStyle(fontWeight: FontWeight.bold, color: bandColor(f['cancer_risk_band'] as String?))),
                ])),
              ),
            if (f['brock_risk'] != null)
              Tooltip(
                message: 'Brock (PanCan) clinical model: age, sex, family history, emphysema + nodule size, type, '
                    'location, count, spiculation',
                child: Text.rich(TextSpan(children: [
                  const TextSpan(text: 'Brock clinical risk '),
                  TextSpan(
                      text: '${pct(f['brock_risk'])} · ${f['brock_band']}',
                      style: TextStyle(fontWeight: FontWeight.bold, color: bandColor(f['brock_band'] as String?))),
                ])),
              )
            else
              Text("Brock risk: enter the patient's age", style: small),
            if (f['needs_review'] == true)
              const Text('⚠ needs your review', style: TextStyle(color: kOrange, fontWeight: FontWeight.w600)),
          ]),
          const SizedBox(height: 8),
          Wrap(spacing: 8, runSpacing: 4, crossAxisAlignment: WrapCrossAlignment.center, children: [
            DropdownButton<String>(
              value: ['solid', 'part-solid', 'nonsolid'].contains(attrs['nodule_type']) ? attrs['nodule_type'] as String : 'solid',
              isDense: true,
              items: const [
                DropdownMenuItem(value: 'solid', child: Text('solid')),
                DropdownMenuItem(value: 'part-solid', child: Text('part-solid')),
                DropdownMenuItem(value: 'nonsolid', child: Text('nonsolid')),
              ],
              onChanged: (t) => _setAttrs(id, {...attrs, 'nodule_type': t}),
            ),
            Row(mainAxisSize: MainAxisSize.min, children: [
              Checkbox(value: attrs['upper_lobe'] == true, onChanged: (b) => _setAttrs(id, {...attrs, 'upper_lobe': b ?? false})),
              const Text('upper lobe'),
            ]),
            Row(mainAxisSize: MainAxisSize.min, children: [
              Checkbox(value: attrs['spiculated'] == true, onChanged: (b) => _setAttrs(id, {...attrs, 'spiculated': b ?? false})),
              const Text('spiculated'),
            ]),
          ]),
          const SizedBox(height: 6),
          Wrap(spacing: 8, runSpacing: 8, crossAxisAlignment: WrapCrossAlignment.center, children: [
            decide('confirm', 'Confirm nodule', Icons.check),
            decide('reject', 'False positive', Icons.close),
            decide('unsure', 'Unsure', Icons.help_outline),
            SizedBox(
              width: 220,
              child: TextField(
                controller: _notes.putIfAbsent(id, () => TextEditingController()),
                decoration: field('note (optional)'),
              ),
            ),
          ]),
        ]),
      ),
    );
  }

  Widget _impressionCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const SectionTitle('Impression'),
          TextField(
            controller: _impression,
            minLines: 4,
            maxLines: 10,
            decoration: field('Your overall impression and recommendation for this study…'),
          ),
          const SizedBox(height: 10),
          Row(children: [
            FilledButton(onPressed: _saveImpression, child: const Text('Save impression')),
            const SizedBox(width: 12),
            Expanded(child: Text(_impMsg, style: Theme.of(context).textTheme.bodySmall)),
          ]),
        ]),
      ),
    );
  }
}

/// Coloured bar with one cell per slice (green / yellow / red) and a marker at the current slice.
class _RiskBarPainter extends CustomPainter {
  final List<double> risk;
  final int z;
  final double yellow, red;
  _RiskBarPainter(this.risk, this.z, this.yellow, this.red);

  @override
  void paint(Canvas canvas, Size size) {
    if (risk.isEmpty) return;
    final w = size.width / risk.length;
    final paint = Paint();
    for (var i = 0; i < risk.length; i++) {
      paint.color = risk[i] >= red ? kRed : (risk[i] >= yellow ? kYellow : kGreen);
      canvas.drawRect(Rect.fromLTWH(i * w, 0, w + 0.6, size.height), paint);
    }
    paint.color = Colors.white;
    canvas.drawRect(Rect.fromLTWH(z * w + w / 2 - 1, 0, 2, size.height), paint);
  }

  @override
  bool shouldRepaint(covariant _RiskBarPainter old) =>
      old.z != z || old.yellow != yellow || old.red != red || old.risk != risk;
}

/// Circles around AI findings on the current slice (size shrinks away from the nodule centre).
class _FindingsPainter extends CustomPainter {
  final List<Map<String, dynamic>> findings;
  final int z;
  final double sx, sz, scale;
  final int? selected;
  final Set<int> rejected;
  _FindingsPainter({
    required this.findings,
    required this.z,
    required this.sx,
    required this.sz,
    required this.scale,
    required this.selected,
    required this.rejected,
  });

  @override
  void paint(Canvas canvas, Size size) {
    for (final f in findings) {
      final id = asInt(f['id']);
      final v = (f['voxel_zyx'] as List).map((e) => asInt(e)).toList();
      final rMm = math.max(3.0, asDouble(f['diameter_mm'])! / 2) + 3;
      final dzMm = (v[0] - z).abs() * sz;
      if (dzMm > rMm) continue;
      final r = math.sqrt(math.max(rMm * rMm - dzMm * dzMm, 4)) / sx * scale;
      final rej = rejected.contains(id);
      final color = rej ? const Color(0xFF8A8A85) : (id == selected ? const Color(0xFFFF9B54) : const Color(0xFFF2C14E));
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;
      final c = Offset(v[2] * scale, v[1] * scale);
      if (rej) {
        // dashed circle for rejected findings
        const segs = 24;
        for (var i = 0; i < segs; i += 2) {
          canvas.drawArc(Rect.fromCircle(center: c, radius: r), i * 2 * math.pi / segs, 2 * math.pi / segs, false, paint);
        }
      } else {
        canvas.drawCircle(c, r, paint);
      }
      final tp = TextPainter(
        text: TextSpan(text: '$id', style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.bold)),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, c + Offset(r + 4, -r - tp.height / 2));
    }
  }

  @override
  bool shouldRepaint(covariant _FindingsPainter old) => true;
}
