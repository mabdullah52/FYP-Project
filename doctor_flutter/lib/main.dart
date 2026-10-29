import 'package:flutter/material.dart';

import 'api.dart';
import 'screens/login.dart';
import 'screens/patients.dart';

void main() {
  runApp(const LungScanDoctorApp());
}

const kBrand = Color(0xFF1F6FD1);

class LungScanDoctorApp extends StatelessWidget {
  const LungScanDoctorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LungScan AI - Clinician workstation',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: kBrand),
        useMaterial3: true,
        visualDensity: VisualDensity.compact,
      ),
      home: const StartGate(),
    );
  }
}

/// Decides between "create administrator", "sign in" and the patient list.
class StartGate extends StatefulWidget {
  const StartGate({super.key});

  @override
  State<StartGate> createState() => _StartGateState();
}

class _StartGateState extends State<StartGate> {
  bool _loading = true;
  bool _setupNeeded = false;
  bool _loggedIn = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _check();
  }

  Future<void> _check() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    await Api.instance.restore();
    try {
      final st = await Api.instance.authState();
      _setupNeeded = st['setup_needed'] == true;
      _loggedIn = st['user'] != null;
    } catch (e) {
      _error = e.toString();
      _loggedIn = false;
    }
    if (mounted) setState(() => _loading = false);
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    if (_loggedIn) return PatientsScreen(onLogout: _check);
    return LoginScreen(setupNeeded: _setupNeeded, initialError: _error, onDone: _check);
  }
}
