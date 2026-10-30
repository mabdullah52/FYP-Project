import 'dart:convert';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

/// Client for the LungScan AI server (doctor_app/server.py), which runs the AI model on the GPU PC.
/// After login every call carries the session token in an "Authorization: Bearer" header.
class ApiException implements Exception {
  final String message;
  final int status;
  ApiException(this.message, [this.status = 0]);
  @override
  String toString() => message;
}

class Api {
  Api._();
  static final Api instance = Api._();

  String baseUrl = '';
  String token = '';
  Map<String, dynamic>? user;

  bool get isAdmin => user?['role'] == 'admin';
  String get userName => (user?['name'] ?? user?['username'] ?? '').toString();

  /// The address of the PC running the AI server. In the browser the app is served by that
  /// server itself, so the default is the page's own address.
  static String defaultServer() {
    if (kIsWeb) return Uri.base.origin;
    return 'http://127.0.0.1:8000';
  }

  Future<void> restore() async {
    final p = await SharedPreferences.getInstance();
    baseUrl = p.getString('server') ?? defaultServer();
    token = p.getString('token') ?? '';
  }

  void setServer(String url) {
    var u = url.trim();
    if (u.isEmpty) u = defaultServer();
    if (!u.startsWith('http')) u = 'http://$u';
    while (u.endsWith('/')) {
      u = u.substring(0, u.length - 1);
    }
    baseUrl = u;
  }

  Future<void> saveSession() async {
    final p = await SharedPreferences.getInstance();
    await p.setString('server', baseUrl);
    await p.setString('token', token);
  }

  /// {setup_needed: bool, user: {...} | null}
  Future<Map<String, dynamic>> authState() async {
    final r = await send('GET', '/api/auth/state') as Map<String, dynamic>;
    user = r['user'] as Map<String, dynamic>?;
    return r;
  }

  Future<void> login(String username, String password) async {
    final r = await send('POST', '/api/auth/login', body: {'username': username, 'password': password});
    token = r['token'] as String;
    user = r['user'] as Map<String, dynamic>;
    await saveSession();
  }

  Future<void> setupAdmin(String username, String name, String password) async {
    final r = await send('POST', '/api/auth/setup', body: {'username': username, 'name': name, 'password': password});
    token = r['token'] as String;
    user = r['user'] as Map<String, dynamic>;
    await saveSession();
  }

  Future<void> logout() async {
    try {
      await send('POST', '/api/auth/logout');
    } catch (_) {}
    token = '';
    user = null;
    await saveSession();
  }

  Map<String, String> get _auth => token.isEmpty ? {} : {'Authorization': 'Bearer $token'};

  Uri uri(String path) => Uri.parse('$baseUrl$path');

  /// Link that opens in a normal browser tab (printable report); the token rides in the URL.
  String linkWithToken(String path) => '$baseUrl$path${path.contains('?') ? '&' : '?'}token=${Uri.encodeComponent(token)}';

  Future<dynamic> send(String method, String path, {Object? body, Duration timeout = const Duration(seconds: 60)}) async {
    final headers = {'Content-Type': 'application/json', ..._auth};
    final req = http.Request(method, uri(path))..headers.addAll(headers);
    if (body != null) req.body = jsonEncode(body);
    http.Response r;
    try {
      r = await http.Response.fromStream(await req.send().timeout(timeout));
    } catch (_) {
      throw ApiException('Cannot reach the AI server at $baseUrl. Is 7_DOCTOR_APP.bat running on the PC, '
          'and is this device on the same network?');
    }
    return _decode(r);
  }

  dynamic _decode(http.Response r) {
    final text = utf8.decode(r.bodyBytes);
    if (r.statusCode >= 300) {
      String msg = 'Error ${r.statusCode}';
      try {
        msg = (jsonDecode(text)['detail'] ?? msg).toString();
      } catch (_) {}
      throw ApiException(msg, r.statusCode);
    }
    if (text.isEmpty) return null;
    return jsonDecode(text);
  }

  /// Raw bytes (CT slice PNGs, heatmaps, snapshots). Returns null for "204 no content".
  Future<Uint8List?> bytes(String path) async {
    final r = await http.get(uri(path), headers: _auth).timeout(const Duration(seconds: 60));
    if (r.statusCode == 204) return null;
    if (r.statusCode >= 300) throw ApiException('Error ${r.statusCode}', r.statusCode);
    return r.bodyBytes;
  }

  /// Upload CT files for a patient. Each file is (name, path) on desktop/Android or (name, bytes) on web.
  /// CT volumes such as .raw are stored uncompressed and are mostly air, so they shrink to about
  /// half (or less) when zipped. Packing them into one .zip before the upload cuts the upload time;
  /// the server unpacks .zip files. Already-compressed files (.mha, .gz, .zip) are sent as they are.
  static const _packable = ['.raw', '.mhd', '.dcm', '.nii', '.nrrd', '.img', '.hdr'];

  Future<List<UploadItem>> _packForUpload(List<UploadItem> files) async {
    final lower = files.map((f) => f.name.toLowerCase()).toList();
    final packable = lower.every((n) => _packable.any(n.endsWith) || !n.contains('.'));
    // only files already in memory (the web app); on Android/Windows the files are streamed from disk
    if (!packable || files.any((f) => f.bytes == null)) return files;
    final archive = Archive();
    for (final f in files) {
      final data = f.bytes!;
      archive.addFile(ArchiveFile(f.name, data.length, data));
    }
    final zipped = ZipEncoder().encode(archive, level: 1);
    if (zipped == null) return files;
    final first = files.first.name;
    final base = first.contains('.') ? first.substring(0, first.lastIndexOf('.')) : first;
    return [UploadItem('$base.zip', bytes: Uint8List.fromList(zipped))];
  }

  Future<int> uploadStudy(int patientId, List<UploadItem> files) async {
    try {
      files = await _packForUpload(files);
    } catch (_) {
      // packing is only a speed-up: if it fails, upload the original files
    }
    final req = http.MultipartRequest('POST', uri('/api/studies/upload'))
      ..headers.addAll(_auth)
      ..fields['patient_id'] = '$patientId';
    for (final f in files) {
      if (f.bytes != null) {
        req.files.add(http.MultipartFile.fromBytes('files', f.bytes!, filename: f.name));
      } else {
        req.files.add(await http.MultipartFile.fromPath('files', f.path!, filename: f.name));
      }
    }
    http.Response r;
    try {
      r = await http.Response.fromStream(await req.send().timeout(const Duration(minutes: 30)));
    } catch (_) {
      throw ApiException('Upload failed: the AI server at $baseUrl could not be reached.');
    }
    return (_decode(r)['id'] as num).toInt();
  }
}

class UploadItem {
  final String name;
  final String? path;
  final Uint8List? bytes;
  UploadItem(this.name, {this.path, this.bytes});
}

// ------------------------------------------------------------------ small helpers used by the screens
double? asDouble(dynamic v) => v == null ? null : (v as num).toDouble();
int asInt(dynamic v) => v == null ? 0 : (v as num).toInt();
String fmtDate(dynamic s) => (s ?? '').toString().replaceFirst('T', ' ');
String pct(dynamic v, [int digits = 1]) => v == null ? '–' : '${((v as num) * 100).toStringAsFixed(digits)}%';
