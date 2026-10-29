import 'package:flutter/material.dart';

import '../api.dart';
import '../widgets/common.dart';

/// New / edit patient dialog, including the lifestyle data used by the Brock clinical risk model.
/// Returns the saved patient, or null when cancelled.
Future<Map<String, dynamic>?> showPatientForm(BuildContext context, Map<String, dynamic>? patient) {
  return showDialog<Map<String, dynamic>>(context: context, builder: (_) => _PatientForm(patient: patient));
}

class _PatientForm extends StatefulWidget {
  final Map<String, dynamic>? patient;
  const _PatientForm({this.patient});

  @override
  State<_PatientForm> createState() => _PatientFormState();
}

class _PatientFormState extends State<_PatientForm> {
  final _name = TextEditingController();
  final _mrn = TextEditingController();
  final _age = TextEditingController();
  final _phone = TextEditingController();
  final _notes = TextEditingController();
  final _pack = TextEditingController();
  String? _sex;
  String? _smoking;
  bool _family = false;
  bool _emphysema = false;
  bool _busy = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    final p = widget.patient;
    if (p != null) {
      _name.text = '${p['name'] ?? ''}';
      _mrn.text = '${p['mrn'] ?? ''}';
      _age.text = p['age'] == null ? '' : '${p['age']}';
      _phone.text = '${p['phone'] ?? ''}';
      _notes.text = '${p['notes'] ?? ''}';
      _pack.text = p['pack_years'] == null ? '' : '${p['pack_years']}';
      _sex = p['sex'] as String?;
      _smoking = p['smoking_status'] as String?;
      _family = p['family_history'] == 1 || p['family_history'] == true;
      _emphysema = p['emphysema'] == 1 || p['emphysema'] == true;
    }
    if (!['Female', 'Male', 'Other', null].contains(_sex)) _sex = null;
    if (!['never', 'former', 'current', null].contains(_smoking)) _smoking = null;
  }

  String? _txt(TextEditingController c) => c.text.trim().isEmpty ? null : c.text.trim();

  Future<void> _save() async {
    if (_name.text.trim().isEmpty) {
      setState(() => _error = 'Name is required.');
      return;
    }
    final body = {
      'name': _name.text.trim(),
      'mrn': _txt(_mrn),
      'age': int.tryParse(_age.text.trim()),
      'sex': _sex,
      'phone': _txt(_phone),
      'notes': _txt(_notes),
      'smoking_status': _smoking,
      'pack_years': double.tryParse(_pack.text.trim()),
      'family_history': _family,
      'emphysema': _emphysema,
    };
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final p = widget.patient;
      final r = p == null
          ? await Api.instance.send('POST', '/api/patients', body: body)
          : await Api.instance.send('PUT', '/api/patients/${p['id']}', body: body);
      if (mounted) Navigator.pop(context, r as Map<String, dynamic>);
    } catch (e) {
      setState(() {
        _error = e.toString();
        _busy = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    const gap = SizedBox(height: 12, width: 12);
    return AlertDialog(
      title: Text(widget.patient == null ? 'New patient' : 'Edit patient'),
      content: SizedBox(
        width: 480,
        child: SingleChildScrollView(
          child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            TextField(controller: _name, decoration: field('Full name *')),
            gap,
            Row(children: [
              Expanded(child: TextField(controller: _mrn, decoration: field('MRN'))),
              gap,
              Expanded(child: TextField(controller: _age, decoration: field('Age'), keyboardType: TextInputType.number)),
              gap,
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _sex,
                  decoration: field('Sex'),
                  items: const [
                    DropdownMenuItem(value: 'Female', child: Text('Female')),
                    DropdownMenuItem(value: 'Male', child: Text('Male')),
                    DropdownMenuItem(value: 'Other', child: Text('Other')),
                  ],
                  onChanged: (v) => setState(() => _sex = v),
                ),
              ),
            ]),
            gap,
            TextField(controller: _phone, decoration: field('Phone'), keyboardType: TextInputType.phone),
            gap,
            TextField(controller: _notes, decoration: field('Notes'), maxLines: 2),
            const SizedBox(height: 16),
            Text('Lifestyle and risk factors (used by the Brock clinical model)',
                style: Theme.of(context).textTheme.titleSmall),
            gap,
            Row(children: [
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _smoking,
                  decoration: field('Smoking'),
                  items: const [
                    DropdownMenuItem(value: 'never', child: Text('Never smoked')),
                    DropdownMenuItem(value: 'former', child: Text('Former smoker')),
                    DropdownMenuItem(value: 'current', child: Text('Current smoker')),
                  ],
                  onChanged: (v) => setState(() => _smoking = v),
                ),
              ),
              gap,
              Expanded(
                child: TextField(
                  controller: _pack,
                  decoration: field('Pack-years'),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
              ),
            ]),
            CheckboxListTile(
              contentPadding: EdgeInsets.zero,
              value: _family,
              onChanged: (v) => setState(() => _family = v ?? false),
              title: const Text('Lung cancer in a parent, brother or sister'),
            ),
            CheckboxListTile(
              contentPadding: EdgeInsets.zero,
              value: _emphysema,
              onChanged: (v) => setState(() => _emphysema = v ?? false),
              title: const Text('Emphysema'),
            ),
            if (_error != null) Text(_error!, style: const TextStyle(color: kRed)),
          ]),
        ),
      ),
      actions: [
        TextButton(onPressed: _busy ? null : () => Navigator.pop(context), child: const Text('Cancel')),
        FilledButton(onPressed: _busy ? null : _save, child: Text(widget.patient == null ? 'Create' : 'Save')),
      ],
    );
  }
}
