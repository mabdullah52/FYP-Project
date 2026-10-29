import 'package:flutter/material.dart';

/// Shared look: text field decoration, status chips, Lung-RADS / risk colours, message helpers.
InputDecoration field(String label, {String? hint}) =>
    InputDecoration(labelText: label, hintText: hint, border: const OutlineInputBorder(), isDense: true);

void toast(BuildContext context, String msg, {bool error = false}) {
  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
    content: Text(msg),
    backgroundColor: error ? const Color(0xFFB3261E) : null,
    duration: Duration(seconds: error ? 6 : 3),
  ));
}

Future<bool> confirmDialog(BuildContext context, String title, String text, {String ok = 'OK'}) async {
  final r = await showDialog<bool>(
    context: context,
    builder: (c) => AlertDialog(
      title: Text(title),
      content: Text(text),
      actions: [
        TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Cancel')),
        FilledButton(onPressed: () => Navigator.pop(c, true), child: Text(ok)),
      ],
    ),
  );
  return r == true;
}

const kGreen = Color(0xFF2E9E4F);
const kYellow = Color(0xFFE0A800);
const kRed = Color(0xFFD03B3A);
const kOrange = Color(0xFFE07A1F);

Color bandColor(String? band) {
  switch (band) {
    case 'high':
      return kRed;
    case 'intermediate':
      return kOrange;
    case 'low':
      return kGreen;
  }
  return Colors.grey;
}

Color radsColor(String cat) {
  if (cat.startsWith('2')) return kGreen;
  if (cat.startsWith('3')) return kYellow;
  if (cat.startsWith('4A')) return kOrange;
  return kRed;
}

class StatusChip extends StatelessWidget {
  final String status;
  const StatusChip(this.status, {super.key});

  @override
  Widget build(BuildContext context) {
    final c = {
          'done': kGreen,
          'error': kRed,
          'processing': kBrandBlue,
          'queued': Colors.grey,
          'uploading': Colors.grey,
        }[status] ??
        Colors.grey;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(color: c.withOpacity(0.12), borderRadius: BorderRadius.circular(10)),
      child: Text(status, style: TextStyle(color: c, fontSize: 12, fontWeight: FontWeight.w600)),
    );
  }
}

const kBrandBlue = Color(0xFF1F6FD1);

class Pill extends StatelessWidget {
  final String text;
  final Color color;
  const Pill(this.text, this.color, {super.key});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
        decoration: BoxDecoration(color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(10)),
        child: Text(text, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600)),
      );
}

/// Section heading with an optional widget on the right.
class SectionTitle extends StatelessWidget {
  final String text;
  final Widget? trailing;
  const SectionTitle(this.text, {this.trailing, super.key});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Row(children: [
          Expanded(child: Text(text, style: Theme.of(context).textTheme.titleMedium)),
          if (trailing != null) trailing!,
        ]),
      );
}
