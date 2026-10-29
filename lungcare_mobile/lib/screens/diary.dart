import 'package:flutter/material.dart';

import '../api.dart';

class DiaryScreen extends StatefulWidget {
  const DiaryScreen({super.key});
  @override
  State<DiaryScreen> createState() => _DiaryScreenState();
}

class _DiaryScreenState extends State<DiaryScreen> {
  final _symptom = TextEditingController();
  final _notes = TextEditingController();
  double _severity = 3;
  List<dynamic> _items = [];
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final d = await Api.instance.home();
      if (mounted) setState(() => _items = (d['symptoms'] ?? []) as List<dynamic>);
    } catch (_) {}
  }

  Future<void> _save() async {
    if (_symptom.text.trim().isEmpty) return;
    setState(() => _busy = true);
    try {
      final r = await Api.instance.logSymptom(_symptom.text.trim(), _severity.round(), _notes.text.trim().isEmpty ? null : _notes.text.trim());
      _symptom.clear();
      _notes.clear();
      if (!mounted) return;
      if (r['urgent'] == true) {
        await showDialog(
          context: context,
          builder: (_) => AlertDialog(
            icon: const Icon(Icons.warning_amber, color: Colors.red, size: 40),
            title: const Text('Get help now'),
            content: Text(r['advice'] ?? 'Please seek emergency care. Your doctor has been alerted.'),
            actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Saved in your diary')));
      }
      _load();
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.toString())));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(padding: const EdgeInsets.all(16), children: [
        Text('How are you feeling?', style: t.textTheme.titleLarge),
        const SizedBox(height: 12),
        TextField(controller: _symptom, decoration: const InputDecoration(labelText: 'Symptom (e.g. cough, tiredness)', border: OutlineInputBorder())),
        const SizedBox(height: 12),
        Text('How bad is it? ${_severity.round()} / 10'),
        Slider(value: _severity, min: 1, max: 10, divisions: 9, label: '${_severity.round()}', onChanged: (v) => setState(() => _severity = v)),
        TextField(controller: _notes, decoration: const InputDecoration(labelText: 'Notes (optional)', border: OutlineInputBorder())),
        const SizedBox(height: 12),
        FilledButton.icon(onPressed: _busy ? null : _save, icon: const Icon(Icons.save), label: const Text('Save to diary')),
        const SizedBox(height: 8),
        Text('Coughing up blood, chest pain, severe breathlessness or fainting are treated as URGENT: your doctor is '
            'alerted and you will be told to seek emergency care.', style: t.textTheme.bodySmall),
        const Divider(height: 32),
        Text('My diary', style: t.textTheme.titleMedium),
        if (_items.isEmpty) const Padding(padding: EdgeInsets.all(8), child: Text('No entries yet.')),
        ..._items.map((s) => Card(
              child: ListTile(
                leading: Icon((s['urgent'] ?? 0) == 1 ? Icons.warning_amber : Icons.notes,
                    color: (s['urgent'] ?? 0) == 1 ? Colors.red : null),
                title: Text('${s['symptom']}${s['severity'] != null ? '  (${s['severity']}/10)' : ''}'),
                subtitle: Text('${(s['at'] ?? '').toString().replaceAll('T', ' ')}${s['notes'] != null ? '\n${s['notes']}' : ''}'),
              ),
            )),
      ]),
    );
  }
}
