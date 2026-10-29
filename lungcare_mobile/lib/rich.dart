import 'package:flutter/material.dart';

/// Renders the assistant's light markdown: **bold**, _italic_, line breaks and "• " bullets.
class RichMessage extends StatelessWidget {
  final String text;
  final TextStyle? style;
  const RichMessage(this.text, {super.key, this.style});

  @override
  Widget build(BuildContext context) {
    final base = style ?? DefaultTextStyle.of(context).style;
    final spans = <TextSpan>[];
    final re = RegExp(r'\*\*(.+?)\*\*|_(.+?)_');
    var last = 0;
    for (final m in re.allMatches(text)) {
      if (m.start > last) spans.add(TextSpan(text: text.substring(last, m.start)));
      if (m.group(1) != null) {
        spans.add(TextSpan(text: m.group(1), style: const TextStyle(fontWeight: FontWeight.w700)));
      } else {
        spans.add(TextSpan(text: m.group(2), style: const TextStyle(fontStyle: FontStyle.italic)));
      }
      last = m.end;
    }
    if (last < text.length) spans.add(TextSpan(text: text.substring(last)));
    return SelectableText.rich(TextSpan(style: base, children: spans));
  }
}
