imports = ['ua-silero-demo', 'ukrainian_tts', 'whisper']

def import_all():
  import pathlib
  p = pathlib.Path().resolve()
  for i in imports:
    raw = rf'{p}\{i}'
    import sys
    sys.path.insert(0, raw)