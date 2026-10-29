import 'package:flutter/material.dart';

import '../api.dart';

class DoctorScreen extends StatefulWidget {
  const DoctorScreen({super.key});
  @override
  State<DoctorScreen> createState() => _DoctorScreenState();
}

class _DoctorScreenState extends State<DoctorScreen> {
  final _text = TextEditingController();
  String _kind = 'question';
  String _urgency = 'routine';
  List<dynamic> _requests = [];
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final d = await Api.instance.home();
      if (mounted) setState(() => _requests = (d['requests'] ?? []) as List<dynamic>);
    } catch (_) {}
  }

  Future<void> _send() async {
    if (_text.text.trim().isEmpty) return;
    setState(() => _busy = true);
    try {
      await Api.instance.sendRequest(_kind, _text.text.trim(), urgency: _urgency);
      _text.clear();
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Sent to your doctor')));
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
        Text('Message your doctor', style: t.textTheme.titleLarge),
        const SizedBox(height: 12),
        SegmentedButton<String>(
          segments: const [
            ButtonSegment(value: 'question', label: Text('Question'), icon: Icon(Icons.help_outline)),
            ButtonSegment(value: 'appointment', label: Text('Appointment'), icon: Icon(Icons.event)),
          ],
          selected: {_kind},
          onSelectionChanged: (s) => setState(() => _kind = s.first),
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _text,
          minLines: 3,
          maxLines: 6,
          decoration: InputDecoration(
            labelText: _kind == 'question' ? 'Your question' : 'Reason and preferred date',
            border: const OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          value: _urgency,
          decoration: const InputDecoration(labelText: 'How soon?', border: OutlineInputBorder()),
          items: const [
            DropdownMenuItem(value: 'routine', child: Text('Routine')),
            DropdownMenuItem(value: 'soon', child: Text('Soon')),
            DropdownMenuItem(value: 'urgent', child: Text('Urgent')),
          ],
          onChanged: (v) => setState(() => _urgency = v ?? 'routine'),
        ),
        const SizedBox(height: 12),
        FilledButton.icon(onPressed: _busy ? null : _send, icon: const Icon(Icons.send), label: const Text('Send')),
        const Divider(height: 32),
        Text('Messages and replies', style: t.textTheme.titleMedium),
        if (_requests.isEmpty) const Padding(padding: EdgeInsets.all(8), child: Text('No messages yet.')),
        ..._requests.map((r) => Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('${r['kind']} · ${(r['created'] ?? '').toString().replaceAll('T', ' ')}', style: t.textTheme.labelSmall),
                  const SizedBox(height: 4),
                  Text(r['text'] ?? ''),
                  const SizedBox(height: 6),
                  if (r['reply'] != null)
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(color: t.colorScheme.secondaryContainer, borderRadius: BorderRadius.circular(8)),
                      child: Text('Doctor${r['replied_by'] != null ? ' (${r['replied_by']})' : ''}: ${r['reply']}'),
                    )
                  else
                    Text('Waiting for a reply', style: t.textTheme.bodySmall),
                ]),
              ),
            )),
      ]),
    );
  }
}
