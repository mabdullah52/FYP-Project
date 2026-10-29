import 'package:flutter/material.dart';

import '../api.dart';

/// Approved guideline library (same documents the assistant searches).
class GuidesScreen extends StatefulWidget {
  const GuidesScreen({super.key});
  @override
  State<GuidesScreen> createState() => _GuidesScreenState();
}

class _GuidesScreenState extends State<GuidesScreen> {
  final _q = TextEditingController();
  List<dynamic> _docs = [];
  List<dynamic>? _results;

  @override
  void initState() {
    super.initState();
    Api.instance.library().then((d) {
      if (mounted) setState(() => _docs = (d['documents'] ?? []) as List<dynamic>);
    }).catchError((_) {});
  }

  Future<void> _search() async {
    if (_q.text.trim().isEmpty) {
      setState(() => _results = null);
      return;
    }
    final d = await Api.instance.library(_q.text.trim());
    setState(() => _results = (d['results'] ?? []) as List<dynamic>);
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return ListView(padding: const EdgeInsets.all(16), children: [
      TextField(
        controller: _q,
        textInputAction: TextInputAction.search,
        onSubmitted: (_) => _search(),
        decoration: InputDecoration(
          hintText: 'Search the guides (e.g. PET scan, exercises)',
          prefixIcon: const Icon(Icons.search),
          suffixIcon: IconButton(icon: const Icon(Icons.clear), onPressed: () {
            _q.clear();
            setState(() => _results = null);
          }),
          border: const OutlineInputBorder(),
        ),
      ),
      const SizedBox(height: 12),
      if (_results != null) ...[
        if (_results!.isEmpty) const Text('Nothing found - ask the assistant or your doctor.'),
        ..._results!.map((r) => Card(
              child: ListTile(
                title: Text(r['section'] ?? ''),
                subtitle: Text('${r['text']}\n\nFrom: ${r['doc']}'),
              ),
            )),
      ] else
        ..._docs.map((d) => Card(
              child: ExpansionTile(
                title: Text(d['title'] ?? ''),
                subtitle: Text(d['source'] ?? '', style: t.textTheme.bodySmall),
                childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                children: [
                  for (final s in (d['sections'] as List<dynamic>))
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text(s['heading'] ?? '', style: t.textTheme.titleSmall),
                        const SizedBox(height: 2),
                        Text(s['text'] ?? ''),
                      ]),
                    ),
                ],
              ),
            )),
      const SizedBox(height: 8),
      Text('These guides are general information. Your doctor\'s advice always comes first.', style: t.textTheme.bodySmall),
    ]);
  }
}
