import 'package:flutter/material.dart';

import '../api.dart';
import '../widgets/common.dart';

/// Administrator: user accounts, AI settings and the activity (audit) log.
class AdminScreen extends StatefulWidget {
  const AdminScreen({super.key});

  @override
  State<AdminScreen> createState() => _AdminScreenState();
}

class _AdminScreenState extends State<AdminScreen> {
  final api = Api.instance;
  List<dynamic> _users = [];
  List<dynamic> _logs = [];
  Map<String, dynamic> _settings = {};
  final Map<String, TextEditingController> _set = {
    'detection_threshold': TextEditingController(),
    'review_uncertainty': TextEditingController(),
    'slice_yellow': TextEditingController(),
    'slice_red': TextEditingController(),
    'max_findings': TextEditingController(),
  };
  bool _fast = true;
  String _setMsg = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final u = await api.send('GET', '/api/admin/users');
      final s = await api.send('GET', '/api/settings');
      final l = await api.send('GET', '/api/admin/audit');
      if (!mounted) return;
      setState(() {
        _users = u as List<dynamic>;
        _settings = s as Map<String, dynamic>;
        _logs = l as List<dynamic>;
        for (final e in _set.entries) {
          e.value.text = '${_settings[e.key] ?? ''}';
        }
        _fast = '${_settings['fast_mode'] ?? 1}' != '0';
      });
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _saveSettings() async {
    final body = <String, dynamic>{for (final e in _set.entries) e.key: double.tryParse(e.value.text.trim())};
    body.removeWhere((k, v) => v == null);
    body['fast_mode'] = _fast ? 1 : 0;
    try {
      await api.send('PUT', '/api/admin/settings', body: body);
      setState(() => _setMsg = 'saved ✓ - applies to scans analysed from now on (Re-analyze older studies)');
      _load();
    } catch (e) {
      setState(() => _setMsg = e.toString());
    }
  }

  Future<void> _patchUser(int id, Map<String, dynamic> body) async {
    try {
      await api.send('PUT', '/api/admin/users/$id', body: body);
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
    _load();
  }

  Future<void> _addUser() async {
    final user = TextEditingController(), name = TextEditingController(), pw = TextEditingController();
    String role = 'doctor';
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => StatefulBuilder(
        builder: (c, setD) => AlertDialog(
          title: const Text('Add user'),
          content: SizedBox(
            width: 380,
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              TextField(controller: user, decoration: field('Username')),
              const SizedBox(height: 10),
              TextField(controller: name, decoration: field('Full name (e.g. Dr. Ayesha Khan)')),
              const SizedBox(height: 10),
              DropdownButtonFormField<String>(
                value: role,
                decoration: field('Role'),
                items: const [
                  DropdownMenuItem(value: 'doctor', child: Text('doctor')),
                  DropdownMenuItem(value: 'admin', child: Text('admin')),
                ],
                onChanged: (v) => setD(() => role = v ?? 'doctor'),
              ),
              const SizedBox(height: 10),
              TextField(controller: pw, obscureText: true, decoration: field('Password (at least 8 characters)')),
            ]),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Add user')),
          ],
        ),
      ),
    );
    if (ok != true) return;
    try {
      await api.send('POST', '/api/admin/users',
          body: {'username': user.text.trim(), 'name': name.text.trim(), 'role': role, 'password': pw.text});
      _load();
    } catch (e) {
      if (mounted) toast(context, e.toString(), error: true);
    }
  }

  Future<void> _resetPassword(int id) async {
    final pw = TextEditingController();
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Reset password'),
        content: TextField(controller: pw, obscureText: true, decoration: field('New password (at least 8 characters)')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Change')),
        ],
      ),
    );
    if (ok == true && pw.text.isNotEmpty) {
      await _patchUser(id, {'password': pw.text});
      if (mounted) toast(context, 'Password changed.');
    }
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Administration'),
          bottom: const TabBar(tabs: [Tab(text: 'Users'), Tab(text: 'AI settings'), Tab(text: 'Activity log')]),
          actions: [IconButton(onPressed: _load, icon: const Icon(Icons.refresh))],
        ),
        body: TabBarView(children: [_usersTab(), _settingsTab(), _logTab()]),
      ),
    );
  }

  Widget _usersTab() {
    return ListView(padding: const EdgeInsets.all(16), children: [
      Row(children: [
        Expanded(child: Text('${_users.length} user(s)', style: Theme.of(context).textTheme.titleMedium)),
        FilledButton.icon(onPressed: _addUser, icon: const Icon(Icons.person_add), label: const Text('Add user')),
      ]),
      const SizedBox(height: 8),
      for (final u0 in _users)
        Builder(builder: (_) {
          final u = u0 as Map<String, dynamic>;
          final id = asInt(u['id']);
          return Card(
            child: ListTile(
              title: Text('${u['name']}  (${u['username']})'),
              subtitle: Text('last login: ${u['last_login'] == null ? 'never' : fmtDate(u['last_login'])}'),
              trailing: Wrap(spacing: 8, crossAxisAlignment: WrapCrossAlignment.center, children: [
                DropdownButton<String>(
                  value: u['role'] == 'admin' ? 'admin' : 'doctor',
                  items: const [
                    DropdownMenuItem(value: 'doctor', child: Text('doctor')),
                    DropdownMenuItem(value: 'admin', child: Text('admin')),
                  ],
                  onChanged: (v) => _patchUser(id, {'role': v}),
                ),
                const Text('active'),
                Switch(value: u['active'] == 1 || u['active'] == true, onChanged: (v) => _patchUser(id, {'active': v})),
                TextButton(onPressed: () => _resetPassword(id), child: const Text('Reset password')),
              ]),
            ),
          );
        }),
    ]);
  }

  Widget _settingsTab() {
    Widget numField(String key, String label) => Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: TextField(
            controller: _set[key],
            decoration: field(label),
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
          ),
        );
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560),
        child: ListView(padding: const EdgeInsets.all(16), children: [
          numField('detection_threshold', 'Detection threshold (nodule probability, 0-1)'),
          numField('review_uncertainty', '"Needs review" when uncertainty above (0-1)'),
          numField('slice_yellow', 'Slice bar: yellow from (0-1)'),
          numField('slice_red', 'Slice bar: red from (0-1)'),
          numField('max_findings', 'Max findings per scan'),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            value: _fast,
            onChanged: (v) => setState(() => _fast = v),
            title: const Text('Fast analysis (recommended)'),
            subtitle: const Text('Quick pass on all candidates, full TTA + uncertainty only for the top ones '
                '(about 4x faster, same accuracy). Off = thorough mode.'),
          ),
          const SizedBox(height: 12),
          Row(children: [
            FilledButton(onPressed: _saveSettings, child: const Text('Save settings')),
            const SizedBox(width: 12),
            Expanded(child: Text(_setMsg, style: Theme.of(context).textTheme.bodySmall)),
          ]),
        ]),
      ),
    );
  }

  Widget _logTab() {
    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: _logs.length,
      separatorBuilder: (_, __) => const Divider(height: 1),
      itemBuilder: (_, i) {
        final l = _logs[i] as Map<String, dynamic>;
        return ListTile(
          dense: true,
          leading: SizedBox(width: 150, child: Text(fmtDate(l['at']), style: Theme.of(context).textTheme.bodySmall)),
          title: Text('${l['user']} · ${l['action']}'),
          subtitle: (l['detail'] ?? '').toString().isEmpty ? null : Text('${l['detail']}'),
        );
      },
    );
  }
}
