import gradio as gr


from deep_translator import MyMemoryTranslator

def translate_text(text):
    return MyMemoryTranslator(source='ar-EG', target='en-GB').translate(text)

print(translate_text("مرحبا كيف حالك؟"))