import 'dart:async';

import 'package:flutter/material.dart';

import '../api.dart';
import '../widgets/common.dart';
import 'admin.dart';
import 'patient.dart';
import 'patient_form.dart';

/// Home screen: patient list with search, model status, admin and logout.
class PatientsScreen extends StatefulWidget {
  final VoidCallback onLogout;
  const PatientsScreen({super.key, required this.onLogout});

  @override
  State<PatientsScreen> createState() => _PatientsScreenState();
}

class _PatientsScreenState extends State<PatientsScreen> {
  final _search = TextEditingController();
  List<dynamic> _patients = [];
  bool _loading = true;
  String? _error;
  Map<String, dynamic>? _health;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _load();
    _loadHealth();
    _timer = Timer.periodic(const Duration(seconds: 15), (_) => _loadHealth());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _load() async {
    try {
      final r = await Api.instance.send('GET', '/api/patients?q=${Uri.encodeQueryComponent(_search.text)}');
      if (!mounted) return;
      setState(() {
        _patients = r as List<dynamic>;
        _loading = false;
        _error = null;
      });
    } on ApiException catch (e) {
      if (e.status == 401) {
        widget.onLogout();
        return;
      }
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _loadHealth() async {
    try {
      final h = await Api.instance.send('GET', '/api/health');
      if (mounted) setState(() => _health = h as Map<String, dynamic>);
    } catch (_) {}
  }

  Widget _modelPill() {
    final h = _health;
    if (h == null) return const SizedBox.shrink();
    if (h['error'] != null) return Tooltip(message: '${h['error']}', child: const Pill('model error', kRed));
    if (h['model'] == null) return const Pill('no trained model', kRed);
    final st = '${h['state']}';
    final ready = st.startsWith('ready');
    return Tooltip(
      message: 'model $st\n${h['model']}',
      child: Pill(ready ? 'AI ready' : 'AI $st', ready ? kGreen : Colors.grey),
    );
  }

  Future<void> _newPatient() async {
    final p = await showPatientForm(context, null);
    if (p != null && mounted) {
      await _load();
      if (mounted) _open(asInt(p['id']));
    }
  }

  Future<void> _open(int id) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => PatientScreen(patientId: id)));
    _load();
  }

  @override
  Widget build(BuildContext context) {
    final api = Api.instance;
    return Scaffold(
      appBar: AppBar(
        title: const Row(children: [
          Icon(Icons.health_and_safety, color: kBrandBlue),
          SizedBox(width: 8),
          Text('LungScan AI'),
        ]),
        actions: [
          _modelPill(),
          const SizedBox(width: 12),
          Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 180),
              child: Text('${api.userName} · ${api.user?['role'] ?? ''}',
                  overflow: TextOverflow.ellipsis, style: Theme.of(context).textTheme.bodySmall),
            ),
          ),
          if (api.isAdmin)
            IconButton(
              tooltip: 'Administration',
              icon: const Icon(Icons.admin_panel_settings_outlined),
              onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const AdminScreen())),
            ),
          IconButton(
            tooltip: 'Log out',
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await api.logout();
              widget.onLogout();
            },
          ),
          const SizedBox(width: 8),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _newPatient,
        icon: const Icon(Icons.person_add_alt_1),
        label: const Text('New patient'),
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 900),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(children: [
              TextField(
                controller: _search,
                decoration: field('Search name or MRN').copyWith(prefixIcon: const Icon(Icons.search)),
                onChanged: (_) => _load(),
              ),
              const SizedBox(height: 12),
              if (_error != null) Text(_error!, style: const TextStyle(color: kRed)),
              Expanded(
                child: _loading
                    ? const Center(child: CircularProgressIndicator())
                    : _patients.isEmpty
                        ? const Center(child: Text('No patients yet. Click "New patient" to add one.'))
                        : RefreshIndicator(
                            onRefresh: _load,
                            child: ListView.separated(
                              itemCount: _patients.length,
                              separatorBuilder: (_, __) => const Divider(height: 1),
                              itemBuilder: (_, i) {
                                final p = _patients[i] as Map<String, dynamic>;
                                final n = asInt(p['n_studies']);
                                final open = asInt(p['n_open']);
                                return ListTile(
                                  leading: CircleAvatar(child: Text(('${p['name']}'.isEmpty ? '?' : '${p['name']}'[0]).toUpperCase())),
                                  title: Text('${p['name']}'),
                                  subtitle: Text([
                                    if (p['mrn'] != null && '${p['mrn']}'.isNotEmpty) 'MRN ${p['mrn']}',
                                    '${p['age'] ?? '?'} y ${p['sex'] ?? ''}',
                                    '$n ${n == 1 ? 'study' : 'studies'}',
                                  ].join(' · ')),
                                  trailing: open > 0
                                      ? Tooltip(
                                          message: 'open patient messages',
                                          child: Badge(label: Text('$open'), child: const Icon(Icons.mail_outline)),
                                        )
                                      : const Icon(Icons.chevron_right),
                                  onTap: () => _open(asInt(p['id'])),
                                );
                              },
                            ),
                          ),
              ),
            ]),
          ),
        ),
      ),
    );
  }
}
