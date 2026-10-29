import 'package:flutter/material.dart';

import 'api.dart';
import 'screens/assistant.dart';
import 'screens/diary.dart';
import 'screens/doctor.dart';
import 'screens/guides.dart';
import 'screens/home.dart';
import 'screens/login.dart';

void main() => runApp(const LungCareApp());

class LungCareApp extends StatelessWidget {
  const LungCareApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF0E7C6B);
    return MaterialApp(
      title: 'LungCare',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: seed, useMaterial3: true),
      darkTheme: ThemeData(colorSchemeSeed: seed, useMaterial3: true, brightness: Brightness.dark),
      home: const Gate(),
    );
  }
}

/// Opens the login screen, or the main app if a code is already saved on the phone.
class Gate extends StatefulWidget {
  const Gate({super.key});
  @override
  State<Gate> createState() => _GateState();
}

class _GateState extends State<Gate> {
  bool? _loggedIn;

  @override
  void initState() {
    super.initState();
    Api.instance.restore().then((ok) => setState(() => _loggedIn = ok));
  }

  @override
  Widget build(BuildContext context) {
    if (_loggedIn == null) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    return _loggedIn!
        ? Shell(onLogout: () => setState(() => _loggedIn = false))
        : LoginScreen(onDone: () => setState(() => _loggedIn = true));
  }
}

class Shell extends StatefulWidget {
  final VoidCallback onLogout;
  const Shell({super.key, required this.onLogout});
  @override
  State<Shell> createState() => _ShellState();
}

class _ShellState extends State<Shell> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      HomeScreen(onAsk: () => setState(() => _tab = 1)),
      const AssistantScreen(),
      const DiaryScreen(),
      const DoctorScreen(),
      const GuidesScreen(),
    ];
    return Scaffold(
      appBar: AppBar(
        title: Text('LungCare · ${Api.instance.patientName.split(' ').first}'),
        actions: [
          IconButton(
            tooltip: 'Log out',
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await Api.instance.logout();
              widget.onLogout();
            },
          ),
        ],
      ),
      body: SafeArea(child: IndexedStack(index: _tab, children: pages)),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home), label: 'Results'),
          NavigationDestination(icon: Icon(Icons.chat_bubble_outline), selectedIcon: Icon(Icons.chat_bubble), label: 'Assistant'),
          NavigationDestination(icon: Icon(Icons.edit_note_outlined), selectedIcon: Icon(Icons.edit_note), label: 'Diary'),
          NavigationDestination(icon: Icon(Icons.local_hospital_outlined), selectedIcon: Icon(Icons.local_hospital), label: 'Doctor'),
          NavigationDestination(icon: Icon(Icons.menu_book_outlined), selectedIcon: Icon(Icons.menu_book), label: 'Guides'),
        ],
      ),
    );
  }
}
