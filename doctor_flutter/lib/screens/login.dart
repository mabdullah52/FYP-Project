import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import '../api.dart';
import '../widgets/common.dart';

/// Sign in, or create the administrator account on the very first start.
class LoginScreen extends StatefulWidget {
  final bool setupNeeded;
  final String? initialError;
  final VoidCallback onDone;
  const LoginScreen({super.key, required this.setupNeeded, required this.onDone, this.initialError});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _server = TextEditingController();
  final _user = TextEditingController();
  final _name = TextEditingController();
  final _pw = TextEditingController();
  bool _busy = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _server.text = Api.instance.baseUrl;
    _error = widget.initialError;
  }

  Future<void> _submit() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    final api = Api.instance;
    api.setServer(_server.text);
    try {
      if (widget.setupNeeded) {
        if (_pw.text.length < 8) throw ApiException('The password needs at least 8 characters.');
        await api.setupAdmin(_user.text.trim(), _name.text.trim(), _pw.text);
      } else {
        await api.login(_user.text.trim(), _pw.text);
      }
      widget.onDone();
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// Re-check the server (e.g. after changing its address): maybe the admin account is still missing.
  Future<void> _recheck() async {
    Api.instance.setServer(_server.text);
    await Api.instance.saveSession();
    widget.onDone();
  }

  void _forgot() {
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Forgot your password?'),
        content: const SizedBox(
          width: 420,
          child: Text(
            'For safety, passwords are never sent by e-mail.\n\n'
            '- Doctors: ask the administrator. In Administration > Users they click '
            '"Reset password" next to your name and tell you the new one.\n\n'
            '- Administrator (or nobody else can help): on the PC that runs the AI server, '
            'double-click RESET_PASSWORD.bat in the project folder, type your username and a new '
            'password. Only someone at that PC can do this.\n\n'
            'Every reset is recorded in the activity log.',
          ),
        ),
        actions: [FilledButton(onPressed: () => Navigator.pop(c), child: const Text('OK'))],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final setup = widget.setupNeeded;
    return Scaffold(
      backgroundColor: const Color(0xFFF3F5F8),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Card(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: AutofillGroup(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(children: [
                        const Icon(Icons.health_and_safety, color: kBrandBlue, size: 30),
                        const SizedBox(width: 8),
                        Text('LungScan AI', style: Theme.of(context).textTheme.headlineSmall),
                      ]),
                      const SizedBox(height: 4),
                      Text('Clinician workstation', style: Theme.of(context).textTheme.bodySmall),
                      const SizedBox(height: 20),
                      Text(setup ? 'Create the administrator account' : 'Sign in',
                          style: Theme.of(context).textTheme.titleLarge),
                      const SizedBox(height: 6),
                      Text(
                        setup
                            ? 'First start: this account can add doctors and change AI settings. Keep the password safe.'
                            : 'Authorised staff only.',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      const SizedBox(height: 16),
                      if (!kIsWeb) ...[
                        TextField(
                          controller: _server,
                          decoration: field('AI server address', hint: 'http://192.168.1.20:8000'),
                          onSubmitted: (_) => _recheck(),
                        ),
                        const SizedBox(height: 4),
                        Align(
                          alignment: Alignment.centerRight,
                          child: TextButton(onPressed: _recheck, child: const Text('Check server')),
                        ),
                        const SizedBox(height: 8),
                      ],
                      TextField(
                        controller: _user,
                        decoration: field('Username'),
                        autofillHints: const [AutofillHints.username],
                      ),
                      if (setup) ...[
                        const SizedBox(height: 12),
                        TextField(controller: _name, decoration: field('Full name (shown on reports)')),
                      ],
                      const SizedBox(height: 12),
                      TextField(
                        controller: _pw,
                        obscureText: true,
                        decoration: field(setup ? 'Password (at least 8 characters)' : 'Password'),
                        autofillHints: const [AutofillHints.password],
                        onSubmitted: (_) => _submit(),
                      ),
                      if (_error != null) ...[
                        const SizedBox(height: 12),
                        Text(_error!, style: const TextStyle(color: kRed)),
                      ],
                      const SizedBox(height: 20),
                      FilledButton(
                        onPressed: _busy ? null : _submit,
                        child: Padding(
                          padding: const EdgeInsets.symmetric(vertical: 10),
                          child: _busy
                              ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                              : Text(setup ? 'Create account' : 'Sign in'),
                        ),
                      ),
                      if (!setup)
                        Align(
                          alignment: Alignment.center,
                          child: TextButton(onPressed: _forgot, child: const Text('Forgot password?')),
                        ),
                      const SizedBox(height: 12),
                      Text(
                        'Research prototype - decision support only, not a medical diagnosis.',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
