import 'dart:convert';

import 'package:http/http.dart' as http;
import 'local_store.dart';

/// Thin client for the LungCare patient server (patient_app/server.py).
/// Every call carries the patient's access code in the X-Code header.
class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}

class Api {
  Api._();
  static final Api instance = Api._();

  String baseUrl = '';
  String code = '';
  String patientName = '';

  /// Restores a saved login. Returns false (= show the login screen) when nothing is saved,
  /// when the phone's storage cannot be read, or when the saved server or code no longer works
  /// (for example an old Wi-Fi address from an earlier version, or a code the clinic replaced).
  Future<bool> restore() async {
    try {
      final p = await LocalStore.instance().timeout(const Duration(seconds: 5));
      baseUrl = p.getString('baseUrl') ?? '';
      code = p.getString('code') ?? '';
      patientName = p.getString('name') ?? '';
    } catch (_) {
      return false;
    }
    if (baseUrl.isEmpty || code.isEmpty) return false;
    try {
      await http
          .get(Uri.parse('$baseUrl/api/home'), headers: {'X-Code': code})
          .timeout(const Duration(seconds: 15))
          .then((r) {
        if (r.statusCode >= 300) throw ApiException('saved login no longer valid');
      });
      return true;
    } catch (_) {
      code = '';
      baseUrl = '';   // the login screen then offers the built-in online address
      return false;
    }
  }

  Future<void> login(String url, String accessCode) async {
    var u = url.trim();
    if (!u.startsWith('http')) u = 'http://$u';
    if (u.endsWith('/')) u = u.substring(0, u.length - 1);
    baseUrl = u;
    final r = await _send('POST', '/api/login', body: {'code': accessCode.trim()}, auth: false);
    code = r['code'] as String;
    patientName = r['name'] as String? ?? '';
    final p = await LocalStore.instance();
    await p.setString('baseUrl', baseUrl);
    await p.setString('code', code);
    await p.setString('name', patientName);
  }

  Future<void> logout() async {
    final p = await LocalStore.instance();
    await p.remove('code');
    code = '';
  }

  Future<dynamic> _send(String method, String path, {Map<String, dynamic>? body, bool auth = true}) async {
    final uri = Uri.parse('$baseUrl$path');
    final headers = {'Content-Type': 'application/json', if (auth) 'X-Code': code};
    http.Response r;
    try {
      switch (method) {
        case 'POST':
          r = await http.post(uri, headers: headers, body: jsonEncode(body ?? {})).timeout(const Duration(seconds: 60));
          break;
        case 'PUT':
          r = await http.put(uri, headers: headers).timeout(const Duration(seconds: 30));
          break;
        case 'DELETE':
          r = await http.delete(uri, headers: headers).timeout(const Duration(seconds: 30));
          break;
        default:
          r = await http.get(uri, headers: headers).timeout(const Duration(seconds: 30));
      }
    } catch (_) {
      throw ApiException('Cannot reach the hospital server at $baseUrl. Check the address and that your phone is '
          'on the same Wi-Fi as the PC running the patient app.');
    }
    final text = utf8.decode(r.bodyBytes);
    if (r.statusCode >= 300) {
      String msg = 'Error ${r.statusCode}';
      try {
        msg = (jsonDecode(text)['detail'] ?? msg).toString();
      } catch (_) {}
      throw ApiException(msg);
    }
    return text.isEmpty ? {} : jsonDecode(text);
  }

  Future<Map<String, dynamic>> home() async => (await _send('GET', '/api/home')) as Map<String, dynamic>;
  Future<List<dynamic>> chatHistory() async => (await _send('GET', '/api/chat')) as List<dynamic>;
  Future<Map<String, dynamic>> chat(String message) async =>
      (await _send('POST', '/api/chat', body: {'message': message})) as Map<String, dynamic>;
  Future<void> addReminder(String title, String dueDate, {String? note}) =>
      _send('POST', '/api/reminders', body: {'title': title, 'due_date': dueDate, 'note': note});
  Future<void> reminderDone(int id, bool done) => _send('PUT', '/api/reminders/$id/done?done=$done');
  Future<void> deleteReminder(int id) => _send('DELETE', '/api/reminders/$id');
  Future<Map<String, dynamic>> logSymptom(String symptom, int? severity, String? notes) async =>
      (await _send('POST', '/api/symptoms', body: {'symptom': symptom, 'severity': severity, 'notes': notes}))
          as Map<String, dynamic>;
  Future<void> sendRequest(String kind, String text, {String urgency = 'routine'}) =>
      _send('POST', '/api/requests', body: {'kind': kind, 'text': text, 'urgency': urgency});
  /// "Forgot your code?" - phone number + MRN recorded by the clinic; returns {code, name}.
  Future<Map<String, dynamic>> recoverCode(String url, String phone, String mrn) async {
    var u = url.trim();
    if (!u.startsWith('http')) u = 'http://$u';
    if (u.endsWith('/')) u = u.substring(0, u.length - 1);
    baseUrl = u;
    return (await _send('POST', '/api/recover', body: {'phone': phone, 'mrn': mrn}, auth: false)) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> library([String q = '']) async =>
      (await _send('GET', '/api/library?q=${Uri.encodeQueryComponent(q)}')) as Map<String, dynamic>;
}
