用于在cosyvoice2时提取语音包

使用方式：使用本项目中的文件替换cosyvoice/cli/cosyvoice.py和cosyvoice/cli/frontend.py

提取语音包示例（pt格式）
sys.path.insert(0, os.path.abspath('third_party/Matcha-TTS'))
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
prompt_speech_16k = load_wav(example.wav, 16000)
model = cosyvoice.get_pt(prompt_text=prompt, prompt_speech_16k = prompt_speech_16k, name='0')
torch.save(model, 'example.pt')


调用语音包生成语音
model = torch.load('example.pt')
for i, j in enumerate(cosyvoice.inference_zero_shot_pt(text, stream=False, voice_input=model)):
    torchaudio.save(f'temp.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
