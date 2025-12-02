import gradio as gr

from modules import shared, ui_prompt_styles
import modules.images

from modules.ui_components import ToolButton


class Toprow:
    """Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation"""

    prompt = None
    prompt_img = None
    negative_prompt = None

    button_interrogate = None
    button_deepbooru = None

    interrupt = None
    interrupting = None
    skip = None
    submit = None

    paste = None
    clear_prompt_button = None
    apply_styles = None
    restore_progress_button = None

    token_counter = None
    token_button = None
    negative_token_counter = None
    negative_token_button = None

    ui_styles = None

    submit_box = None

    def __init__(self, is_img2img, is_compact=False, id_part=None):
        if id_part is None:
            id_part = "img2img" if is_img2img else "txt2img"

        self.id_part = id_part
        self.is_img2img = is_img2img
        self.is_compact = is_compact

        if not is_compact:
            with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
                self.create_classic_toprow()
        else:
            self.create_submit_box()

    def create_classic_toprow(self):
        self.create_prompts()

        with gr.Column(scale=1, elem_id=f"{self.id_part}_actions_column"):
            self.create_submit_box()

            self.create_tools_row()

            self.create_styles_ui()

    def create_inline_toprow_prompts(self):
        if not self.is_compact:
            return

        self.create_prompts()

        with gr.Row(elem_classes=["toprow-compact-stylerow"]):
            with gr.Column(elem_classes=["toprow-compact-tools"]):
                self.create_tools_row()
            with gr.Column():
                self.create_styles_ui()

    def create_inline_toprow_image(self):
        if not self.is_compact:
            return

        self.submit_box.render()

    def create_prompts(self):
        with gr.Column(elem_id=f"{self.id_part}_prompt_container", elem_classes=["prompt-container-compact"] if self.is_compact else [], scale=6):
            with gr.Row(elem_id=f"{self.id_part}_prompt_row", elem_classes=["prompt-row"]):
                self.prompt = gr.Textbox(label="النص", elem_id=f"{self.id_part}_prompt", show_label=False, lines=3, placeholder="النص\n(اضغط Ctrl+Enter للتوليد، Alt+Enter للتخطي، Esc للإيقاف)", elem_classes=["prompt"])
                self.prompt_img = gr.File(label="", elem_id=f"{self.id_part}_prompt_image", file_count="single", type="binary", visible=False)

            with gr.Row(elem_id=f"{self.id_part}_neg_prompt_row", elem_classes=["prompt-row"]):
                self.negative_prompt = gr.Textbox(label="النص السلبي", elem_id=f"{self.id_part}_neg_prompt", show_label=False, lines=3, placeholder="النص السلبي\n(اضغط Ctrl+Enter للتوليد، Alt+Enter للتخطي، Esc للإيقاف)", elem_classes=["prompt"])

        self.prompt_img.change(
            fn=modules.images.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress=False,
        )

    def create_submit_box(self):
        with gr.Row(elem_id=f"{self.id_part}_generate_box", elem_classes=["generate-box"] + (["generate-box-compact"] if self.is_compact else []), render=not self.is_compact) as submit_box:
            self.submit_box = submit_box

            self.interrupt = gr.Button('إيقاف', elem_id=f"{self.id_part}_interrupt", elem_classes="generate-box-interrupt", tooltip="إيقاف التوليد فورًا أو بعد إنهاء الدفعة الحالية")
            self.skip = gr.Button('تخطي', elem_id=f"{self.id_part}_skip", elem_classes="generate-box-skip", tooltip="إيقاف توليد الدفعة الحالية والاستمرار إلى الدفعة التالية")
            self.interrupting = gr.Button('جارٍ الإيقاف...', elem_id=f"{self.id_part}_interrupting", elem_classes="generate-box-interrupting", tooltip="جارٍ إيقاف التوليد...")
            self.submit = gr.Button('توليد', elem_id=f"{self.id_part}_generate", variant='primary', tooltip="انقر بالزر الأيمن لخيارات التوليد المستمر")

            def interrupt_function():
                if not shared.state.stopping_generation and shared.state.job_count > 1 and shared.opts.interrupt_after_current:
                    shared.state.stop_generating()
                    gr.Info("سيتم إيقاف التوليد بعد انتهاء هذه الصورة، انقر مرة أخرى لإيقاف فوري.")
                else:
                    shared.state.interrupt()

            self.skip.click(fn=shared.state.skip)
            self.interrupt.click(fn=interrupt_function, _js='function(){ showSubmitInterruptingPlaceholder("' + self.id_part + '"); }')
            self.interrupting.click(fn=interrupt_function)

    def create_tools_row(self):
        with gr.Row(elem_id=f"{self.id_part}_tools"):
            from modules.ui import paste_symbol, clear_prompt_symbol, restore_progress_symbol

            self.paste = ToolButton(value=paste_symbol, elem_id="paste", tooltip="قراءة إعدادات التوليد من النص أو من آخر توليد (إذا كان النص فارغًا) إلى واجهة المستخدم.")
            self.clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{self.id_part}_clear_prompt", tooltip="مسح النص")
            self.apply_styles = ToolButton(value=ui_prompt_styles.styles_materialize_symbol, elem_id=f"{self.id_part}_style_apply", tooltip="تطبيق جميع الأنماط المحددة على النصوص.")

            if self.is_img2img:
                self.button_interrogate = ToolButton('📎', tooltip='تحليل CLIP - استخدام شبكة CLIP العصبية لإنشاء وصف نصّي للصورة ووضعه في حقل النص.', elem_id="interrogate")
                self.button_deepbooru = ToolButton('📦', tooltip='تحليل DeepBooru - استخدام شبكة DeepBooru العصبية لإنشاء وصف نصّي للصورة ووضعه في حقل النص.', elem_id="deepbooru")

            self.restore_progress_button = ToolButton(value=restore_progress_symbol, elem_id=f"{self.id_part}_restore_progress", visible=False, tooltip="استعادة التقدم")

            self.token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{self.id_part}_token_counter", elem_classes=["token-counter"], visible=False)
            self.token_button = gr.Button(visible=False, elem_id=f"{self.id_part}_token_button")
            self.negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{self.id_part}_negative_token_counter", elem_classes=["token-counter"], visible=False)
            self.negative_token_button = gr.Button(visible=False, elem_id=f"{self.id_part}_negative_token_button")

            self.clear_prompt_button.click(
                fn=lambda *x: x,
                _js="confirm_clear_prompt",
                inputs=[self.prompt, self.negative_prompt],
                outputs=[self.prompt, self.negative_prompt],
            )

    def create_styles_ui(self):
        self.ui_styles = ui_prompt_styles.UiPromptStyles(self.id_part, self.prompt, self.negative_prompt)
        self.ui_styles.setup_apply_button(self.apply_styles)
