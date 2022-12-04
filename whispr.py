import whisper

model = whisper.load_model("base")



def transcribe(path):
  result = model.transcribe(path)
  return result["text"]
