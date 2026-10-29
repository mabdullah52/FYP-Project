import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api.dart';
import '../widgets/common.dart';
import 'patient_form.dart';
import 'study.dart';

/// One patient: details and lifestyle, CT studies, add a new CT, and messages from the patient app.
class PatientScreen extends StatefulWidget {
  final int patientId;
  const PatientScreen({super.key, required this.patientId});

  @override
  State<PatientScreen> createState() => _PatientScreenState();
}

class _PatientScreenState extends State<PatientScreen> {
  Map<String, dynamic>? _p;
  List<dynamic> _requests = [];
  String? _error;
  bool _uploading = false;
  final _path = TextEditingController();
  final Map<int, TextEditingController> _replies = {};

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final p = await Api.instance.send('GET', '/api/patients/${widget.patientId}');
      final r = await Api.instance.send('GET', '/api/patients/${widget.patientId}/requests');
      if (!mounted) return;
      setState(() {
        _p = p as Map<String, dynamic>;
        _requests = r as List<dynamic>;
        _error = null;
      });
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  Future<void> _openStudy(int id) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => StudyScreen(studyId: id)));
    _load();
  }

  Future<void> _pickAndUpload() async {
    final res = await FilePicker.platform.pickFiles(allowMultiple: true, withData: kIsWeb);
    if (res == null || res.files.isEmpty) return;
    final names = res.files.map((f) => f.name.toLowerCase()).toList();
    if (names.any((n) => n.endsWith('.mhd')) && !names.any((n) => n.endsWith('.raw') || n.endsWith('.zraw'))) {
      if (mounted) toast(context, 'Select the .mhd AND its .raw file together.', error: true);
      return;
    }
    final items = res.files
        .map((f) => kIsWeb ? UploadItem(f.name, bytes: f.bytes) : UploadItem(f.name, path: f.path))
        .toList();
    setState(() => _uploading = true);
    try {
      final id = await Api.instance.uploadStudy(widget.patientId, items);
      if (mounted) _openStudy(id);
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    } finally {
      if (mounted) setState(() => _uploading = false);
    }
  }

  Future<void> _fromPath() async {
    final path = _path.text.trim();
    if (path.isEmpty) return;
    try {
      final r = await Api.instance.send('POST', '/api/studies/from_path', body: {'patient_id': widget.patientId, 'path': path});
      _path.clear();
      if (mounted) _openStudy(asInt(r['id']));
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _reply(int rid) async {
    final text = _replies[rid]?.text.trim() ?? '';
    if (text.isEmpty) return;
    try {
      await Api.instance.send('PUT', '/api/requests/$rid', body: {'reply': text});
      _replies[rid]?.clear();
      _load();
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _deletePatient() async {
    final ok = await confirmDialog(context, 'Delete patient',
        'Delete ${_p!['name']} with all CT studies, reviews and messages? This cannot be undone.', ok: 'Delete');
    if (!ok) return;
    try {
      await Api.instance.send('DELETE', '/api/admin/patients/${widget.patientId}');
      if (mounted) Navigator.pop(context);
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _newCode() async {
    final ok = await confirmDialog(context, 'New access code',
        'Give ${_p!['name']} a new patient-app code? The old code stops working immediately (use this if the code was lost or shared).',
        ok: 'New code');
    if (!ok) return;
    try {
      final r = await Api.instance.send('POST', '/api/patients/${widget.patientId}/new-code');
      await _load();
      if (mounted) toast(context, 'New code: ${r['access_code']}');
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  String _yesNo(dynamic v) => v == null ? 'unknown' : (v == 1 || v == true ? 'yes' : 'no');

  @override
  Widget build(BuildContext context) {
    final p = _p;
    return Scaffold(
      appBar: AppBar(
        title: Text(p == null ? 'Patient' : '${p['name']}'),
        actions: [
          if (p != null)
            TextButton.icon(
              onPressed: () async {
                final r = await showPatientForm(context, p);
                if (r != null) _load();
              },
              icon: const Icon(Icons.edit_outlined),
              label: const Text('Edit'),
            ),
          if (p != null && Api.instance.isAdmin)
            IconButton(tooltip: 'Delete patient', icon: const Icon(Icons.delete_outline), onPressed: _deletePatient),
          IconButton(tooltip: 'Refresh', icon: const Icon(Icons.refresh), onPressed: _load),
        ],
      ),
      body: p == null
          ? Center(child: _error != null ? Text(_error!, style: const TextStyle(color: kRed)) : const CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  Center(
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 1000),
                      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                        _infoCard(p),
                        const SizedBox(height: 16),
                        _addCtCard(),
                        const SizedBox(height: 16),
                        _studiesCard(p),
                        if (_requests.isNotEmpty) ...[const SizedBox(height: 16), _requestsCard()],
                      ]),
                    ),
                  ),
                ],
              ),
            ),
    );
  }

  Widget _infoCard(Map<String, dynamic> p) {
    final meta = [
      if (p['mrn'] != null && '${p['mrn']}'.isNotEmpty) 'MRN ${p['mrn']}',
      if (p['age'] != null) '${p['age']} years',
      if (p['sex'] != null) '${p['sex']}',
      if (p['phone'] != null && '${p['phone']}'.isNotEmpty) '${p['phone']}',
    ].join(' · ');
    final smoking = '${p['smoking_status'] ?? 'unknown'}${p['pack_years'] != null ? ', ${p['pack_years']} pack-years' : ''}';
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(meta.isEmpty ? 'No details' : meta, style: Theme.of(context).textTheme.bodyLarge),
          if (p['notes'] != null && '${p['notes']}'.isNotEmpty) ...[const SizedBox(height: 4), Text('${p['notes']}')],
          const SizedBox(height: 8),
          Text('Smoking: $smoking · Family history: ${_yesNo(p['family_history'])} · Emphysema: ${_yesNo(p['emphysema'])}'),
          const SizedBox(height: 12),
          Wrap(crossAxisAlignment: WrapCrossAlignment.center, children: [
            const Icon(Icons.key_outlined, size: 18),
            const SizedBox(width: 6),
            const Text('Patient app access code: '),
            SelectableText('${p['access_code']}', style: const TextStyle(fontWeight: FontWeight.bold, letterSpacing: 2)),
            const SizedBox(width: 8),
            IconButton(
              tooltip: 'Copy code',
              icon: const Icon(Icons.copy, size: 18),
              onPressed: () {
                Clipboard.setData(ClipboardData(text: '${p['access_code']}'));
                toast(context, 'Code copied');
              },
            ),
            TextButton.icon(onPressed: _newCode, icon: const Icon(Icons.autorenew, size: 18), label: const Text('New code')),
          ]),
          Text('Patients who forget it can also recover it in the app with their phone number + MRN.',
              style: Theme.of(context).textTheme.bodySmall),
        ]),
      ),
    );
  }

  Widget _addCtCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const SectionTitle('Add a CT scan'),
          const Text('Upload .mhd + .raw (select both), .mha, NIfTI, a .zip, or DICOM slices. '
              'The AI analyses it in about a minute.'),
          const SizedBox(height: 12),
          Wrap(spacing: 12, runSpacing: 12, crossAxisAlignment: WrapCrossAlignment.center, children: [
            FilledButton.icon(
              onPressed: _uploading ? null : _pickAndUpload,
              icon: _uploading
                  ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.upload_file),
              label: Text(_uploading ? 'Uploading…' : 'Choose CT files'),
            ),
            SizedBox(
              width: 420,
              child: TextField(
                controller: _path,
                decoration: field('…or a path on the AI server PC', hint: r'D:\LUNA16\subset9\1.3.6...mhd'),
                onSubmitted: (_) => _fromPath(),
              ),
            ),
            OutlinedButton(onPressed: _fromPath, child: const Text('Add from path')),
          ]),
        ]),
      ),
    );
  }

  Widget _studiesCard(Map<String, dynamic> p) {
    final studies = (p['studies'] as List<dynamic>?) ?? [];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const SectionTitle('CT studies'),
          if (studies.isEmpty) const Text('No studies yet - add a CT above.'),
          for (final s0 in studies) _studyTile(s0 as Map<String, dynamic>),
        ]),
      ),
    );
  }

  Widget _studyTile(Map<String, dynamic> s) {
    final sm = (s['report'] as Map<String, dynamic>?)?['summary'] as Map<String, dynamic>?;
    final done = s['status'] == 'done';
    final largest = asDouble(sm?['largest_nodule_mm']) ?? 0;
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: const Icon(Icons.view_in_ar_outlined),
      title: Row(children: [
        Text('Study #${s['id']}'),
        const SizedBox(width: 8),
        StatusChip('${s['status']}'),
        if (s['released'] == 1) ...[const SizedBox(width: 8), const Pill('released', kGreen)],
      ]),
      subtitle: Text([
        fmtDate(s['created']),
        if (done) '${sm?['n_nodules_detected'] ?? 0} nodule(s)',
        if (done && largest > 0) 'largest ${largest.toStringAsFixed(1)} mm',
        (s['impression'] ?? '').toString().isNotEmpty ? 'reviewed by ${s['reviewed_by'] ?? '?'}' : 'review pending',
      ].join(' · ')),
      trailing: const Icon(Icons.chevron_right),
      onTap: () => _openStudy(asInt(s['id'])),
    );
  }

  static const _kinds = {'appointment': 'Appointment request', 'question': 'Question', 'symptom': 'Symptom report'};

  Widget _requestsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const SectionTitle('Messages from the patient app'),
          for (final r0 in _requests) _requestTile(r0 as Map<String, dynamic>),
        ]),
      ),
    );
  }

  Widget _requestTile(Map<String, dynamic> r) {
    final id = asInt(r['id']);
    final urgent = r['urgency'] == 'urgent';
    final ctl = _replies.putIfAbsent(id, () => TextEditingController());
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        border: Border.all(color: urgent && r['status'] == 'open' ? kRed : const Color(0xFFDDDDDD)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Row(children: [
          Text(_kinds[r['kind']] ?? '${r['kind']}', style: const TextStyle(fontWeight: FontWeight.bold)),
          const Spacer(),
          if (urgent) const Pill('URGENT', kRed) else Text('${r['urgency']}', style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(width: 8),
          Text(fmtDate(r['created']), style: Theme.of(context).textTheme.bodySmall),
        ]),
        const SizedBox(height: 6),
        Text('${r['text']}'),
        const SizedBox(height: 6),
        if (r['reply'] != null)
          Text('↳ ${r['reply']} — ${r['replied_by'] ?? ''}', style: const TextStyle(color: kBrandBlue))
        else
          Row(children: [
            Expanded(child: TextField(controller: ctl, decoration: field('Reply to the patient…'), onSubmitted: (_) => _reply(id))),
            const SizedBox(width: 8),
            FilledButton(onPressed: () => _reply(id), child: const Text('Send')),
          ]),
      ]),
    );
  }
}
