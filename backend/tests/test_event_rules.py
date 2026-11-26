from __future__ import annotations

import json
import re
from dataclasses import replace

import app.main as main
import app.security as security
import app.vlm_api as api
import app.vlm_jobs as jobs
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch, local_vlm_models):
    monkeypatch.setattr(api, 'DATA_ROOT', tmp_path / 'vlm')
    monkeypatch.setattr(security, 'settings', replace(security.settings, api_token='secret'))
    with TestClient(main.app, headers={'authorization': 'Bearer secret'}) as client:
        yield client


def rule(identifier='custom', **changes):
    return {'id': identifier, 'name': '测试规则', 'description': '是否强行通过', **changes}


def save(client, rules, revision=None, deletes=None):
    if revision is None:
        revision = client.get('/api/vlm/event-rules').json()['revision']
    return client.post('/api/vlm/event-rules/batch', json={'revision': revision, 'upserts': rules, 'deletes': deletes or []})


def test_persistence_batch_validation_and_conflict(client):
    original = client.get('/api/vlm/event-rules').json()
    assert original['rules']
    response = save(client, [rule('one'), rule('two', enabled=False)])
    assert response.status_code == 200, response.text
    saved = response.json()
    assert json.loads((api.DATA_ROOT / 'event-rules.json').read_text()) == saved
    # A mixed valid/invalid batch must write nothing.
    assert save(client, [rule('three'), rule('broken', description=' ')]).status_code == 422
    assert client.get('/api/vlm/event-rules').json() == saved
    assert save(client, [rule('one', description='changed')], revision=original['revision']).status_code == 409
    assert save(client, [rule('one'), rule('one')]).status_code == 422
    assert client.get('/api/vlm/event-rules').json() == saved
    updated = save(client, [rule('one', name='重命名')]).json()
    assert next(r for r in updated['rules'] if r['id'] == 'one')['version'] == 2
    assert save(client, [], deletes=['unknown']).status_code == 404
    assert client.get('/api/vlm/event-rules').json() == updated
    assert save(client, [rule('bad', poiIds=['old-poi'])]).status_code == 422
    deleted = save(client, [], deletes=[r['id'] for r in updated['rules']]).json()
    assert deleted['rules'] == []
    assert client.get('/api/vlm/event-rules').json()['rules'] == []  # Do not re-seed a deliberately empty library.


def start_payload(library, identifier='custom', **changes):
    item = next(r for r in library['rules'] if r['id'] == identifier)
    return {'videoId': 'clip', 'revision': library['revision'],
            'selections': [{'id': identifier, 'version': item['version'], 'poiIds': ['gate']}],
            'pois': [{'id': 'gate', 'kind': 'line', 'points': [[.1, .5], [.9, .5]]}], **changes}


def test_run_uses_backend_definition_and_keeps_snapshot_after_edit_delete(client, monkeypatch):
    library = save(client, [rule()]).json()
    folder = api.DATA_ROOT / 'uploads/clip'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'video.json', {'fileName': 'clip.mp4', 'sourceName': 'source.mp4', 'sha256': 'abc'})
    (folder / 'source.mp4').write_bytes(b'test')

    def launch(run_dir, python):
        # The snapshot must be on disk before the worker sees the run.
        assert (run_dir / 'rules-snapshot.json').exists()
        status = jobs.read_json(run_dir / 'status.json')
        jobs.write_json(run_dir / 'status.json', {**status, 'state': 'completed'})
    monkeypatch.setattr(jobs, 'launch', launch)
    response = client.post('/api/vlm/runs/from-rules', json=start_payload(library))
    assert response.status_code == 200, response.text
    rid = response.json()['runId']
    # Task display name keeps the original video name plus a minute-precision stamp.
    assert re.match(r'clip\.mp4_\d{8}_\d{4}$', client.get(f'/api/vlm/runs/{rid}').json()['displayName'])
    snapshot = client.get(f'/api/vlm/runs/{rid}/rules').json()
    assert snapshot['rules'][0]['version'] == 1
    assert snapshot['rules'][0]['event']['poiIds'] == ['gate']
    request = jobs.read_json(api.DATA_ROOT / f'runs/{rid}/request.json')
    assert request['request']['events'][0]['description'] == '是否强行通过'
    save(client, [rule(description='changed', name='new name')])
    save(client, [], deletes=['custom'])
    assert client.get(f'/api/vlm/runs/{rid}/rules').json() == snapshot
    assert client.get(f'/api/library/vlm/{rid}').json()['ruleSnapshot'] == snapshot
    # A stale selection is rejected rather than silently picking up a newer rule.
    assert client.post('/api/vlm/runs/from-rules', json=start_payload(library)).status_code == 409


def test_cue_binding_flows_to_run_and_unknown_cue_rejected(client, monkeypatch):
    library = save(client, [rule()]).json()
    folder = api.DATA_ROOT / 'uploads/clip'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'video.json', {'fileName': 'clip.mp4', 'sourceName': 'source.mp4', 'sha256': 'abc'})
    (folder / 'source.mp4').write_bytes(b'test')

    def launch(run_dir, _python):
        status = jobs.read_json(run_dir / 'status.json')
        jobs.write_json(run_dir / 'status.json', {**status, 'state': 'completed'})
    monkeypatch.setattr(jobs, 'launch', launch)

    payload = start_payload(library)
    payload['selections'][0]['cueIds'] = ['red']
    payload['cues'] = [{'id': 'red', 'points': [[.2, .2], [.8, .2], [.8, .8], [.2, .8]]}]
    response = client.post('/api/vlm/runs/from-rules', json=payload)
    assert response.status_code == 200, response.text
    request = jobs.read_json(api.DATA_ROOT / f'runs/{response.json()["runId"]}/request.json')['request']
    assert request['events'][0]['cueIds'] == ['red']
    assert request['cues'][0]['id'] == 'red'

    payload = start_payload(library)
    payload['selections'][0]['cueIds'] = ['missing']
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422


def test_empty_cue_binding_expands_to_all_run_cues(client, monkeypatch):
    library = save(client, [rule()]).json()
    folder = api.DATA_ROOT / 'uploads/clip'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'video.json', {'fileName': 'clip.mp4', 'sourceName': 'source.mp4', 'sha256': 'abc'})
    (folder / 'source.mp4').write_bytes(b'test')

    def launch(run_dir, _python):
        status = jobs.read_json(run_dir / 'status.json')
        jobs.write_json(run_dir / 'status.json', {**status, 'state': 'completed'})
    monkeypatch.setattr(jobs, 'launch', launch)
    payload = start_payload(library)
    payload['cues'] = [{'id': 'red', 'points': [[.2, .2], [.8, .2], [.8, .8]]},
                       {'id': 'red2', 'points': [[.1, .1], [.9, .1], [.9, .9]]}]
    payload['selections'][0]['cueIds'] = []
    response = client.post('/api/vlm/runs/from-rules', json=payload)
    assert response.status_code == 200, response.text
    request = jobs.read_json(api.DATA_ROOT / f'runs/{response.json()["runId"]}/request.json')['request']
    assert request['events'][0]['cueIds'] == ['red', 'red2']


def forced_payload(library, identifier='forced'):
    item = next(r for r in library['rules'] if r['id'] == identifier)
    return {
        'videoId': 'clip', 'revision': library['revision'],
        'selections': [{'id': identifier, 'version': item['version'],
                        'poiIds': ['line1', 'zone1'], 'cueIds': ['cue1']}],
        'pois': [
            {'id': 'line1', 'kind': 'line', 'points': [[.1, .5], [.9, .5]],
             'pairIndex': 1, 'entryDirection': 'negative_to_positive'},
            {'id': 'zone1', 'kind': 'zone', 'points': [[.2, .2], [.8, .2], [.8, .8], [.2, .8]],
             'pairIndex': 1},
        ],
        'cues': [{'id': 'cue1', 'points': [[.35, .35], [.65, .35], [.65, .65], [.35, .65]],
                  'pairIndex': 1}],
    }


def test_forced_passage_resolves_paired_line_signal_and_auxiliary_zone(client, monkeypatch):
    library = save(client, [rule('forced', category='forced_passage', triggerKinds=['cue.on'], preSeconds=8)]).json()
    folder = api.DATA_ROOT / 'uploads/clip'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'video.json', {'fileName': 'clip.mp4', 'sourceName': 'source.mp4', 'sha256': 'abc'})
    (folder / 'source.mp4').write_bytes(b'test')

    def launch(run_dir, _python):
        status = jobs.read_json(run_dir / 'status.json')
        jobs.write_json(run_dir / 'status.json', {**status, 'state': 'completed'})
    monkeypatch.setattr(jobs, 'launch', launch)
    response = client.post('/api/vlm/runs/from-rules', json=forced_payload(library))
    assert response.status_code == 200, response.text
    rid = response.json()['runId']
    event = jobs.read_json(api.DATA_ROOT / f'runs/{rid}/request.json')['request']['events'][0]
    assert event['category'] == 'forced_passage'
    assert event['triggerKinds'] == ['cue.on']
    assert event['poiIds'] == ['line1', 'zone1']
    assert event['cueIds'] == ['cue1']
    assert event['preSeconds'] == 8
    snapshot = client.get(f'/api/vlm/runs/{rid}/rules').json()['rules'][0]['event']
    assert snapshot['category'] == 'forced_passage'


def test_forced_passage_binding_validation(client, monkeypatch):
    library = save(client, [rule('forced', category='forced_passage', triggerKinds=['cue.on'])]).json()
    def no_launch(*_args):
        pytest.fail('Invalid forced-passage bindings must not launch a worker')
    monkeypatch.setattr(jobs, 'launch', no_launch)

    payload = forced_payload(library)
    payload['cues'] = []
    payload['selections'][0]['cueIds'] = []
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422

    payload = forced_payload(library)
    payload['cues'][0]['pairIndex'] = 2
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422

    payload = forced_payload(library)
    payload['pois'][0]['entryDirection'] = None
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422

    payload = forced_payload(library)
    payload['pois'] = [payload['pois'][1]]
    payload['selections'][0]['poiIds'] = ['zone1']
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422

    payload = forced_payload(library)
    payload['pois'].append({'id': 'zone2', 'kind': 'zone', 'points': [[.2, .2], [.8, .2], [.8, .8]],
                            'pairIndex': 4})
    payload['selections'][0]['poiIds'].append('zone2')
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422

    # A poi-event rule cannot run when only integrity zones are drawn.
    library = save(client, [rule('plain')]).json()
    payload = integrity_payload(library, identifier='plain')
    payload['selections'][0]['poiIds'] = []
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422


def test_rule_library_category_migration(client):
    jobs.write_json(api.DATA_ROOT / 'event-rules.json', {'revision': 1, 'rules': [
        {'id': 'old-line', 'name': '旧计数', 'description': '计数', 'enabled': True,
         'triggerKinds': ['line.cross'], 'preSeconds': 4, 'postSeconds': 6,
         'criteria': [], 'uncertainWhen': [], 'version': 1, 'createdAt': 0, 'updatedAt': 0},
        {'id': 'old-wall', 'name': '旧完整性', 'description': '完整性', 'enabled': True,
         'triggerKinds': ['integrity.change'], 'preSeconds': 4, 'postSeconds': 6,
         'criteria': [], 'uncertainWhen': [], 'version': 1, 'createdAt': 0, 'updatedAt': 0},
    ]})
    rows = {row['id']: row for row in client.get('/api/vlm/event-rules').json()['rules']}
    assert rows['old-line']['category'] == 'poi_event'
    assert rows['old-wall']['category'] == 'integrity'
    assert rows['preset_integrity']['category'] == 'integrity'


def test_old_rule_without_cueids_still_resolves(client, monkeypatch):
    old_rule = {'id': 'old-rule', 'name': '旧规则', 'description': '是否异常', 'enabled': True,
                'poiIds': [], 'triggerKinds': ['line.cross'], 'preSeconds': 4, 'postSeconds': 6,
                'criteria': [], 'uncertainWhen': [], 'version': 1, 'createdAt': 0, 'updatedAt': 0}
    jobs.write_json(api.DATA_ROOT / 'event-rules.json', {'revision': 1, 'rules': [old_rule]})
    folder = api.DATA_ROOT / 'uploads/clip'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'video.json', {'fileName': 'clip.mp4', 'sourceName': 'source.mp4', 'sha256': 'abc'})
    (folder / 'source.mp4').write_bytes(b'test')

    def launch(run_dir, _python):
        status = jobs.read_json(run_dir / 'status.json')
        jobs.write_json(run_dir / 'status.json', {**status, 'state': 'completed'})
    monkeypatch.setattr(jobs, 'launch', launch)

    payload = {'videoId': 'clip', 'revision': 1,
               'selections': [{'id': 'old-rule', 'version': 1, 'poiIds': ['gate'], 'cueIds': []}],
               'pois': [{'id': 'gate', 'kind': 'line', 'points': [[.1, .5], [.9, .5]]}]}
    response = client.post('/api/vlm/runs/from-rules', json=payload)
    assert response.status_code == 200, response.text
    request = jobs.read_json(api.DATA_ROOT / f'runs/{response.json()["runId"]}/request.json')['request']
    assert request['events'][0]['cueIds'] == []


def test_invalid_bindings_disabled_and_oversized_selection(client):
    library = save(client, [rule()]).json()
    payload = start_payload(library)
    payload['selections'][0]['poiIds'] = ['deleted-zone']
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422
    payload = start_payload(library)
    payload['selections'] *= 31
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422
    library = save(client, [rule(triggerKinds=['zone.enter'])]).json()
    assert client.post('/api/vlm/runs/from-rules', json=start_payload(library)).status_code == 422
    library = save(client, [rule(enabled=False)]).json()
    assert client.post('/api/vlm/runs/from-rules', json=start_payload(library)).status_code == 409


def test_old_snapshot_and_auth(client):
    folder = api.DATA_ROOT / 'runs/old'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'request.json', {'request': {'events': [{'id': 'old', 'description': 'old definition'}]}})
    value = client.get('/api/vlm/runs/old/rules').json()
    assert value['rules'][0]['version'] is None
    assert value['rules'][0]['event']['description'] == 'old definition'
    assert TestClient(main.app).get('/api/vlm/event-rules').status_code == 401
    assert TestClient(main.app).post('/api/vlm/event-rules/batch', json={'revision': 0}).status_code == 401
    assert client.get('/api/vlm/runs/..%2foutside/rules').status_code in (400, 404)


def test_corrupt_library_preserved(client):
    client.get('/api/vlm/event-rules')
    path = api.DATA_ROOT / 'event-rules.json'
    path.write_text('broken')
    assert client.get('/api/vlm/event-rules').status_code == 500
    assert path.read_text() == 'broken'


def test_parameter_defaults_and_custom_values_reach_worker_snapshot(client, monkeypatch):
    contract = client.get('/api/vlm/parameters').json()
    assert contract['defaults'] == jobs.AnalysisParameters().model_dump()
    assert contract['fields']['maxFrames']['multipleOf'] == 2
    assert contract['fields']['conf']['exclusiveMinimum'] == 0
    assert contract['fields']['conf']['exclusiveMaximum'] == 1
    library = save(client, [rule()]).json()
    folder = api.DATA_ROOT / 'uploads/clip'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'video.json', {'fileName': 'clip.mp4', 'sourceName': 'source.mp4', 'sha256': 'abc'})
    (folder / 'source.mp4').write_bytes(b'test')
    settings = {'modelPath': contract['models'][1]['path'], 'poseDevice': 'cpu', 'poseFps': 15, 'conf': .4, 'iou': .6, 'stableSeconds': .4,
                'hysteresisPx': 10, 'maxGapSeconds': 2, 'sampleFps': 4, 'maxFrames': 48,
                'maxPixels': 393216, 'maxNewTokens': 256, 'previewMode': 'lazy', 'dryRun': False}

    def launch(run_dir, _python):
        # Check the exact persisted request consumed by the real worker.
        config = jobs.read_json(run_dir / 'request.json')
        request = config['request']
        assert config['modelPath'] == settings['modelPath']
        assert jobs.read_json(run_dir / 'manifest.json')['configuration']['modelPath'] == settings['modelPath']
        assert {key: request[key] for key in settings} == settings
        assert request['events'][0]['preSeconds'] == 4
        assert request['events'][0]['postSeconds'] == 6
        status = jobs.read_json(run_dir / 'status.json')
        jobs.write_json(run_dir / 'status.json', {**status, 'state': 'completed'})
    monkeypatch.setattr(jobs, 'launch', launch)
    response = client.post('/api/vlm/runs/from-rules', json=start_payload(library, **settings))
    assert response.status_code == 200, response.text
    rid = response.json()['runId']
    assert client.get(f'/api/vlm/runs/{rid}').json()['parameters'] == settings
    assert client.get('/api/vlm/parameters').json()['defaults'] == contract['defaults']


@pytest.mark.parametrize('key,value', [('sampleFps', 0), ('maxFrames', 5), ('maxFrames', 66),
                                     ('maxNewTokens', 127), ('maxPixels', 1), ('conf', 0), ('iou', 1),
                                     ('poseFps', 31), ('maxGapSeconds', .1), ('stableSeconds', -1)])
def test_invalid_parameters_rejected_before_launch(client, monkeypatch, key, value):
    library = save(client, [rule()]).json()
    def no_launch(*_args):
        pytest.fail('Invalid parameters must not launch a worker')
    monkeypatch.setattr(jobs, 'launch', no_launch)
    response = client.post('/api/vlm/runs/from-rules', json=start_payload(library, **{key: value}))
    assert response.status_code == 422
    assert not (api.DATA_ROOT / 'runs').exists()


def integrity_payload(library, identifier='wall-rule', **changes):
    item = next(r for r in library['rules'] if r['id'] == identifier)
    return {'videoId': 'clip', 'revision': library['revision'],
            'selections': [{'id': identifier, 'version': item['version'], 'poiIds': [], 'integrityIds': ['wall']}],
            'pois': [],
            'integrityZones': [{'id': 'wall', 'points': [[.2, .2], [.8, .2], [.8, .8], [.2, .8]]}], **changes}


def test_integrity_only_run_resolves_without_pois(client, monkeypatch):
    library = save(client, [rule('wall-rule', category='integrity', triggerKinds=['integrity.change'])]).json()
    folder = api.DATA_ROOT / 'uploads/clip'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'video.json', {'fileName': 'clip.mp4', 'sourceName': 'source.mp4', 'sha256': 'abc'})
    (folder / 'source.mp4').write_bytes(b'test')

    def launch(run_dir, _python):
        status = jobs.read_json(run_dir / 'status.json')
        jobs.write_json(run_dir / 'status.json', {**status, 'state': 'completed'})
    monkeypatch.setattr(jobs, 'launch', launch)
    response = client.post('/api/vlm/runs/from-rules', json=integrity_payload(library))
    assert response.status_code == 200, response.text
    request = jobs.read_json(api.DATA_ROOT / f'runs/{response.json()["runId"]}/request.json')['request']
    assert request['pois'] == []
    assert request['integrityZones'][0]['id'] == 'wall'
    assert request['integrityZones'][0]['sampleFps'] == 0.5
    assert request['events'][0]['integrityIds'] == ['wall']
    assert request['events'][0]['triggerKinds'] == ['integrity.change']


def test_integrity_binding_validation(client, monkeypatch):
    library = save(client, [rule('wall-rule', category='integrity', triggerKinds=['integrity.change'])]).json()
    def no_launch(*_args):
        pytest.fail('Invalid integrity bindings must not launch a worker')
    monkeypatch.setattr(jobs, 'launch', no_launch)
    payload = integrity_payload(library)
    payload['selections'][0]['integrityIds'] = ['missing']
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422
    # No POI and no integrity zone at all is not a runnable analysis.
    payload = integrity_payload(library)
    payload['integrityZones'] = []
    payload['selections'][0]['integrityIds'] = []
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422
    # An integrity-only trigger cannot bind to a line POI alone.
    payload = integrity_payload(library)
    payload['pois'] = [{'id': 'gate', 'kind': 'line', 'points': [[.1, .5], [.9, .5]]}]
    payload['integrityZones'] = []
    payload['selections'][0]['integrityIds'] = []
    assert client.post('/api/vlm/runs/from-rules', json=payload).status_code == 422
    # Rule templates must not carry integrity bindings.
    assert save(client, [rule('bad', integrityIds=['wall'])]).status_code == 422


def test_preset_integrity_seeded_auto_bound_and_skipped_without_zones(client, monkeypatch):
    library = save(client, [rule()]).json()
    preset = next(r for r in library['rules'] if r['id'] == 'preset_integrity')
    assert preset['triggerKinds'] == ['integrity.change'] and preset['enabled']
    folder = api.DATA_ROOT / 'uploads/clip'
    folder.mkdir(parents=True)
    jobs.write_json(folder / 'video.json', {'fileName': 'clip.mp4', 'sourceName': 'source.mp4', 'sha256': 'abc'})
    (folder / 'source.mp4').write_bytes(b'test')

    def launch(run_dir, _python):
        status = jobs.read_json(run_dir / 'status.json')
        jobs.write_json(run_dir / 'status.json', {**status, 'state': 'completed'})
    monkeypatch.setattr(jobs, 'launch', launch)

    payload = start_payload(library)
    payload['integrityZones'] = [{'id': 'wall', 'points': [[.2, .2], [.8, .2], [.8, .8]]}]
    payload['selections'].append({'id': 'preset_integrity', 'version': preset['version'], 'poiIds': [], 'cueIds': [], 'integrityIds': []})
    response = client.post('/api/vlm/runs/from-rules', json=payload)
    assert response.status_code == 200, response.text
    rid = response.json()['runId']
    request = jobs.read_json(api.DATA_ROOT / f'runs/{rid}/request.json')['request']
    preset_event = next(e for e in request['events'] if e['id'] == 'preset_integrity')
    assert preset_event['integrityIds'] == ['wall']
    snapshot = client.get(f'/api/vlm/runs/{rid}/rules').json()
    assert any(r['id'] == 'preset_integrity' for r in snapshot['rules'])

    # Without integrity zones the preset selection is inert: no preset event runs.
    payload = start_payload(library)
    payload['selections'].append({'id': 'preset_integrity', 'version': preset['version'], 'poiIds': [], 'cueIds': [], 'integrityIds': []})
    response = client.post('/api/vlm/runs/from-rules', json=payload)
    assert response.status_code == 200, response.text
    request = jobs.read_json(api.DATA_ROOT / f'runs/{response.json()["runId"]}/request.json')['request']
    assert [e['id'] for e in request['events']] == ['custom']


def test_legacy_forced_passage_template_is_converted(client):
    jobs.write_json(api.DATA_ROOT / 'event-rules.json', {'revision': 2, 'rules': [
        {'id': 'forced_passage_unopposed', 'name': '旧强行通过', 'description': '是否该人员强行通过安检门且无人阻拦',
         'enabled': True, 'triggerKinds': ['line.cross', 'zone.enter', 'zone.leave'],
         'preSeconds': 2.0, 'postSeconds': 5.0, 'criteria': [], 'uncertainWhen': [],
         'version': 2, 'createdAt': 0, 'updatedAt': 0},
    ]})
    value = client.get('/api/vlm/event-rules').json()
    row = next(r for r in value['rules'] if r['id'] == 'forced_passage_unopposed')
    assert row['category'] == 'forced_passage'
    assert row['triggerKinds'] == ['cue.on']
    assert row['preSeconds'] == 8
    assert row['postSeconds'] == 6
    assert row['version'] == 3
    assert value['revision'] == 3
    # Idempotent: a second read must not bump the version or revision again.
    again = client.get('/api/vlm/event-rules').json()
    assert again['revision'] == 3
    assert next(r for r in again['rules'] if r['id'] == 'forced_passage_unopposed')['version'] == 3
