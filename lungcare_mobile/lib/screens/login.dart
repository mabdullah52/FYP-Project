import 'package:flutter/material.dart';

import '../api.dart';

class LoginScreen extends StatefulWidget {
  final VoidCallback onDone;
  const LoginScreen({super.key, required this.onDone});
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

// Built-in server address: set at build time with --dart-define=LUNGCARE_SERVER=https://...
const _defaultServer = String.fromEnvironment('LUNGCARE_SERVER', defaultValue: 'https://lungscan-ai.thankfulocean-bfa8eccd.koreacentral.azurecontainerapps.io/patient');

class _LoginScreenState extends State<LoginScreen> {
  final _url = TextEditingController(text: Api.instance.baseUrl.isEmpty ? _defaultServer : Api.instance.baseUrl);
  final _code = TextEditingController();
  bool _busy = false;
  String? _error;

  Future<void> _go() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await Api.instance.login(_url.text, _code.text);
      widget.onDone();
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// Forgot the code: phone number + MRN (hospital number) the clinic recorded -> the code is filled in.
  Future<void> _forgot() async {
    final phone = TextEditingController();
    final mrn = TextEditingController();
    String? err;
    bool busy = false;
    final code = await showDialog<String>(
      context: context,
      builder: (c) => StatefulBuilder(
        builder: (c, setD) => AlertDialog(
          title: const Text('Forgot your code?'),
          content: SingleChildScrollView(
            child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              const Text('Enter the phone number and the MRN (hospital number) that the clinic has on record.'),
              const SizedBox(height: 12),
              TextField(
                controller: phone,
                keyboardType: TextInputType.phone,
                decoration: const InputDecoration(labelText: 'Phone number', border: OutlineInputBorder()),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: mrn,
                textCapitalization: TextCapitalization.characters,
                decoration: const InputDecoration(labelText: 'MRN / hospital number', border: OutlineInputBorder()),
              ),
              if (err != null) ...[
                const SizedBox(height: 10),
                Text(err!, style: TextStyle(color: Theme.of(c).colorScheme.error)),
              ],
              const SizedBox(height: 10),
              const Text('If this does not work, call the clinic - they can tell you your code or give you a new one.',
                  style: TextStyle(fontSize: 12)),
            ]),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(c), child: const Text('Cancel')),
            FilledButton(
              onPressed: busy
                  ? null
                  : () async {
                      setD(() {
                        busy = true;
                        err = null;
                      });
                      try {
                        final r = await Api.instance.recoverCode(_url.text, phone.text, mrn.text);
                        if (c.mounted) Navigator.pop(c, r['code'] as String);
                      } catch (e) {
                        setD(() {
                          err = e.toString();
                          busy = false;
                        });
                      }
                    },
              child: const Text('Find my code'),
            ),
          ],
        ),
      ),
    );
    if (code != null && mounted) {
      setState(() => _code.text = code);
      showDialog<void>(
        context: context,
        builder: (c) => AlertDialog(
          title: const Text('Your access code'),
          content: Column(mainAxisSize: MainAxisSize.min, children: [
            SelectableText(code, style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, letterSpacing: 4)),
            const SizedBox(height: 8),
            const Text('It has been filled in for you. The app will remember it after you log in.', textAlign: TextAlign.center),
          ]),
          actions: [FilledButton(onPressed: () => Navigator.pop(c), child: const Text('OK'))],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                Icon(Icons.air, size: 56, color: t.colorScheme.primary),
                const SizedBox(height: 8),
                Text('LungCare', textAlign: TextAlign.center, style: t.textTheme.headlineMedium),
                const SizedBox(height: 4),
                Text('Your CT results, reviewed by your doctor', textAlign: TextAlign.center, style: t.textTheme.bodyMedium),
                const SizedBox(height: 28),
                TextField(
                  controller: _url,
                  keyboardType: TextInputType.url,
                  decoration: const InputDecoration(
                    labelText: 'Hospital server address',
                    helperText: 'Already filled in - only change it if the hospital gives you another address',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _code,
                  textCapitalization: TextCapitalization.characters,
                  decoration: const InputDecoration(
                    labelText: 'Your access code',
                    helperText: 'Given to you by your doctor (6 characters) - you only enter it once',
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => _go(),
                ),
                if (_error != null) ...[
                  const SizedBox(height: 12),
                  Text(_error!, style: TextStyle(color: t.colorScheme.error)),
                ],
                const SizedBox(height: 20),
                FilledButton(
                  onPressed: _busy ? null : _go,
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: _busy ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Open my results'),
                  ),
                ),
                const SizedBox(height: 6),
                TextButton(onPressed: _busy ? null : _forgot, child: const Text('Forgot your code?')),
                const SizedBox(height: 18),
                Text('This app shows results your doctor has reviewed. It does not diagnose. '
                    'If you feel very unwell, cough up blood, have chest pain or cannot breathe, get emergency help now.',
                    style: t.textTheme.bodySmall, textAlign: TextAlign.center),
              ]),
            ),
          ),
        ),
      ),
    );
  }
}
