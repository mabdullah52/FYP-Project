import 'package:flutter/material.dart';

import '../api.dart';
import '../rich.dart';

class AssistantScreen extends StatefulWidget {
  const AssistantScreen({super.key});
  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final _msgs = <Map<String, dynamic>>[];
  final _input = TextEditingController();
  final _scroll = ScrollController();
  bool _busy = false;
  static const _suggestions = [
    'What did my scan show?',
    'What happens next?',
    'Remind me about my next scan',
    'Breathing exercises',
    'What should I ask my doctor?',
    'How do I quit smoking?',
  ];

  @override
  void initState() {
    super.initState();
    Api.instance.chatHistory().then((h) {
      if (!mounted) return;
      setState(() => _msgs.addAll(h.cast<Map<String, dynamic>>()));
      _toEnd();
    }).catchError((_) {});
  }

  void _toEnd() => WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
      });

  Future<void> _send([String? text]) async {
    final msg = (text ?? _input.text).trim();
    if (msg.isEmpty || _busy) return;
    _input.clear();
    setState(() {
      _msgs.add({'role': 'user', 'content': msg});
      _busy = true;
    });
    _toEnd();
    try {
      final r = await Api.instance.chat(msg);
      setState(() => _msgs.add({'role': 'assistant', 'content': r['reply'], 'actions': r['actions']}));
    } catch (e) {
      setState(() => _msgs.add({'role': 'assistant', 'content': e.toString()}));
    } finally {
      if (mounted) setState(() => _busy = false);
      _toEnd();
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(children: [
      Expanded(
        child: ListView.builder(
          controller: _scroll,
          padding: const EdgeInsets.all(12),
          itemCount: _msgs.length + 1,
          itemBuilder: (_, i) {
            if (i == 0) {
              return Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text('Hi! I can explain your results, set reminders, keep your symptom diary, send messages to '
                      'your doctor and share approved health guides. I never diagnose. You can write in English or Urdu.'),
                ),
              );
            }
            final m = _msgs[i - 1];
            final mine = m['role'] == 'user';
            final actions = (m['actions'] ?? []) as List<dynamic>;
            return Align(
              alignment: mine ? Alignment.centerRight : Alignment.centerLeft,
              child: Container(
                margin: const EdgeInsets.symmetric(vertical: 4),
                padding: const EdgeInsets.all(12),
                constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.82),
                decoration: BoxDecoration(
                  color: mine ? cs.primary : cs.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  RichMessage(m['content'] ?? '', style: TextStyle(color: mine ? cs.onPrimary : cs.onSurface)),
                  for (final a in actions)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Chip(label: Text('✓ $a', style: const TextStyle(fontSize: 12)), visualDensity: VisualDensity.compact),
                    ),
                ]),
              ),
            );
          },
        ),
      ),
      if (_busy) const LinearProgressIndicator(),
      SizedBox(
        height: 44,
        child: ListView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 8),
          children: [
            for (final s in _suggestions)
              Padding(padding: const EdgeInsets.symmetric(horizontal: 4), child: ActionChip(label: Text(s), onPressed: () => _send(s))),
          ],
        ),
      ),
      Padding(
        padding: const EdgeInsets.fromLTRB(12, 4, 12, 12),
        child: Row(children: [
          Expanded(
            child: TextField(
              controller: _input,
              minLines: 1,
              maxLines: 4,
              textInputAction: TextInputAction.send,
              onSubmitted: (_) => _send(),
              decoration: const InputDecoration(hintText: 'Type a message…', border: OutlineInputBorder()),
            ),
          ),
          const SizedBox(width: 8),
          IconButton.filled(onPressed: _busy ? null : () => _send(), icon: const Icon(Icons.send)),
        ]),
      ),
    ]);
  }
}
