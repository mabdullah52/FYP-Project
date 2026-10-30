import 'dart:convert';
import 'dart:io';

/// Tiny key-value store in a JSON file inside the app's private folder.
/// Replaces the shared_preferences plugin, whose Android part crashed in release builds
/// ("NoSuchMethodError ... SharedPreferencesApiRegistrar.getAll"). Pure Dart: no plugin needed.
class LocalStore {
  LocalStore._(this._file, this._data);
  final File _file;
  final Map<String, dynamic> _data;

  static LocalStore? _instance;

  static Future<LocalStore> instance() async {
    if (_instance != null) return _instance!;
    // Directory.systemTemp is the app's private cache folder (/data/user/0/<app>/cache);
    // its parent is the app's private data folder, which survives restarts and updates.
    final dir = Directory('${Directory.systemTemp.parent.path}/app_flutter');
    await dir.create(recursive: true);
    final file = File('${dir.path}/lungcare_login.json');
    var data = <String, dynamic>{};
    try {
      if (await file.exists()) data = Map<String, dynamic>.from(jsonDecode(await file.readAsString()) as Map);
    } catch (_) {
      data = {};
    }
    return _instance = LocalStore._(file, data);
  }

  String? getString(String key) => _data[key] as String?;

  Future<void> setString(String key, String value) async {
    _data[key] = value;
    await _save();
  }

  Future<void> remove(String key) async {
    _data.remove(key);
    await _save();
  }

  Future<void> _save() async {
    final tmp = File('${_file.path}.tmp');
    await tmp.writeAsString(jsonEncode(_data), flush: true);
    await tmp.rename(_file.path);
  }
}
