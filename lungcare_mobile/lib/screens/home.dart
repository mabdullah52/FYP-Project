import 'package:flutter/material.dart';

import '../api.dart';
import '../rich.dart';

class HomeScreen extends StatefulWidget {
  final VoidCallback onAsk;
  const HomeScreen({super.key, required this.onAsk});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Map<String, dynamic>? _data;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final d = await Api.instance.home();
      if (mounted) setState(() {
        _data = d;
        _error = null;
      });
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  Color _catColor(String cat, ColorScheme cs) {
    if (cat.startsWith('4')) return const Color(0xFFD03B3A);
    if (cat.startsWith('3')) return const Color(0xFFB88000);
    return const Color(0xFF2E8B57);
  }

  Future<void> _remind(String date) async {
    try {
      await Api.instance.addReminder('Follow-up CT scan', date);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Reminder set')));
      _load();
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.toString())));
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    if (_error != null) {
      return Center(child: Padding(padding: const EdgeInsets.all(24), child: Column(mainAxisSize: MainAxisSize.min, children: [
        Text(_error!, textAlign: TextAlign.center),
        const SizedBox(height: 12),
        OutlinedButton(onPressed: _load, child: const Text('Try again')),
      ])));
    }
    if (_data == null) return const Center(child: CircularProgressIndicator());
    final d = _data!;
    final plan = (d['plan'] ?? {}) as Map<String, dynamic>;
    final studies = (d['studies'] ?? []) as List<dynamic>;
    final reminders = (d['reminders'] ?? []) as List<dynamic>;
    final findings = studies.isNotEmpty ? (studies.first['findings'] as List<dynamic>) : <dynamic>[];
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(padding: const EdgeInsets.all(16), children: [
        Card(child: Padding(padding: const EdgeInsets.all(16), child: RichMessage(d['summary'] ?? ''))),
        if (plan['category_by_size'] != null)
          Card(
            color: t.colorScheme.secondaryContainer,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('What happens next', style: t.textTheme.titleMedium),
                const SizedBox(height: 6),
                Text(plan['general_guidance'] ?? ''),
                const SizedBox(height: 10),
                Row(children: [
                  const Icon(Icons.event, size: 20),
                  const SizedBox(width: 6),
                  Expanded(child: Text('Suggested next scan: ${plan['suggested_next_scan_date']}',
                      style: const TextStyle(fontWeight: FontWeight.w600))),
                ]),
                Text('(${plan['date_source']})', style: t.textTheme.bodySmall),
                const SizedBox(height: 6),
                Text(plan['important'] ?? '', style: t.textTheme.bodySmall),
                const SizedBox(height: 8),
                Wrap(spacing: 8, children: [
                  FilledButton.tonal(onPressed: () => _remind(plan['suggested_next_scan_date']), child: const Text('Remind me')),
                  OutlinedButton(onPressed: widget.onAsk, child: const Text('Ask the assistant')),
                ]),
              ]),
            ),
          ),
        if (findings.isNotEmpty) ...[
          const SizedBox(height: 8),
          Text('Spots your doctor reported', style: t.textTheme.titleMedium),
          ...findings.asMap().entries.map((e) {
            final f = e.value as Map<String, dynamic>;
            final cat = (f['lung_rads_size_category'] ?? '').toString();
            return Card(
              child: ListTile(
                leading: CircleAvatar(backgroundColor: _catColor(cat, t.colorScheme), child: Text('${e.key + 1}',
                    style: const TextStyle(color: Colors.white))),
                title: Text('${(f['diameter_mm'] as num).toStringAsFixed(1)} mm'),
                subtitle: Text('Category ${cat.split(' ').first} (by size)'
                    '${f['doctor_note'] != null ? '\nDoctor: ${f['doctor_note']}' : ''}'),
              ),
            );
          }),
          Text('10 mm = 1 cm. Categories are estimated from size only; your doctor\'s plan always comes first.',
              style: t.textTheme.bodySmall),
        ],
        const SizedBox(height: 12),
        Text('My reminders', style: t.textTheme.titleMedium),
        if (reminders.isEmpty) const Padding(padding: EdgeInsets.all(8), child: Text('No reminders yet.')),
        ...reminders.map((r) => Card(
              child: CheckboxListTile(
                value: (r['done'] ?? 0) == 1,
                title: Text(r['title'] ?? ''),
                subtitle: Text(r['due_date'] ?? ''),
                secondary: IconButton(
                  icon: const Icon(Icons.delete_outline),
                  onPressed: () async {
                    await Api.instance.deleteReminder(r['id'] as int);
                    _load();
                  },
                ),
                onChanged: (v) async {
                  await Api.instance.reminderDone(r['id'] as int, v ?? false);
                  _load();
                },
              ),
            )),
        const SizedBox(height: 16),
        Text('This app shows results your doctor has reviewed. It does not diagnose. If you cough up blood, have chest '
            'pain or cannot breathe, get emergency help now.', style: t.textTheme.bodySmall),
      ]),
    );
  }
}
