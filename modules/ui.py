import datetime
import mimetypes
import os
import sys
from functools import reduce
import warnings
from contextlib import ExitStack

import gradio as gr
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin  # noqa: F401
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call, wrap_gradio_call_no_job # noqa: F401

from modules import gradio_extensons, sd_schedulers  # noqa: F401
from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, extra_networks, ui_common, ui_postprocessing, progress, ui_loadsave, shared_items, ui_settings, timer, sysinfo, ui_checkpoint_merger, scripts, sd_samplers, processing, ui_extra_networks, ui_toprow, launch_utils
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML, InputAccordion, ResizeHandleRow
from modules.paths import script_path
from modules.ui_common import create_refresh_button
from modules.ui_gradio_extensions import reload_javascript

from modules.shared import opts, cmd_opts

import modules.infotext_utils as parameters_copypaste
import modules.hypernetworks.ui as hypernetworks_ui
import modules.textual_inversion.ui as textual_inversion_ui
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.shared as shared
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.infotext_utils import image_from_url_text, PasteField

create_setting_component = ui_settings.create_setting_component

warnings.filterwarnings("default" if opts.show_warnings else "ignore", category=UserWarning)
warnings.filterwarnings("default" if opts.show_gradio_deprecation_warnings else "ignore", category=gr.deprecation.GradioDeprecationWarning)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')

# Likewise, add explicit content-type header for certain missing image types
mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/avif', '.avif')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_options
        )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
restore_progress_symbol = '\U0001F300' # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐


plaintext_to_html = ui_common.plaintext_to_html


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.calculate_target_resolution()

    return f"من <span class='resolution'>{p.width}x{p.height}</span> إلى <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)

    if not target_width or not target_height:
        return "لم يتم اختيار صورة"

    return f"تغيير الحجم: من <span class='resolution'>{width}x{height}</span> إلى <span class='resolution'>{target_width}x{target_height}</span>"


def process_interrogate(interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, "تم التشغيل مع --hide-ui-dir-config، تم تعطيل معالجة img2img الجماعية"
        images = shared.listfiles(ii_input_dir)
        print(f"سيتم معالجة {len(images)} صورة.")
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8'))

        return [gr.update(), None]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def connect_clear_prompt(button):
    """إعداد حدث النقر على زر مسح النص"""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


def update_token_counter(text, steps, styles, *, is_positive=True):
    params = script_callbacks.BeforeTokenCounterParams(text, steps, styles, is_positive=is_positive)
    script_callbacks.before_token_counter_callback(params)
    text = params.prompt
    steps = params.steps
    styles = params.styles
    is_positive = params.is_positive

    if shared.opts.include_styles_into_token_counters:
        apply_styles = shared.prompt_styles.apply_styles_to_prompt if is_positive else shared.prompt_styles.apply_negative_styles_to_prompt
        text = apply_styles(text, styles)

    try:
        text, _ = extra_networks.parse_prompt(text)

        if is_positive:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        else:
            prompt_flat_list = [text]

        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def update_negative_prompt_token_counter(*args):
    return update_token_counter(*args, is_positive=False)


def setup_progressbar(*args, **kwargs):
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()

    if shared.cmd_opts.freeze_settings:
        return gr.update()

    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()

    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)

        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()

    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return

    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()

    opts.save(shared.config_filename)
    return getattr(opts, key)


def create_output_panel(tabname, outdir, toprow=None):
    return ui_common.create_output_panel(tabname, outdir, toprow)


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder_list)}

    for _, category in sorted(enumerate(shared_items.ui_reorder_categories()), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="تجاوز الإعدادات", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=bool(x)),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown


def create_ui():
    import modules.img2img
    import modules.txt2img

    reload_javascript()

    parameters_copypaste.reset()

    settings = ui_settings.UiSettings()
    settings.register_settings()

    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        
       
        toprow = ui_toprow.Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box)

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        toprow = ui_toprow.Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box)

        dummy_component = gr.Label(visible=False)

        extra_tabs = gr.Tabs(elem_id="txt2img_extra_tabs", elem_classes=["extra-networks"])
        extra_tabs.__enter__()

        with gr.Tab("التوليد", id="txt2img_generation") as txt2img_generation_tab, ResizeHandleRow(equal_height=False):
            with ExitStack() as stack:
                if shared.opts.txt2img_settings_accordion:
                    stack.enter_context(gr.Accordion("فتح الإعدادات", open=False))
                stack.enter_context(gr.Column(variant='compact', elem_id="txt2img_settings"))


                






                scripts.scripts_txt2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="txt2img_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="العرض", value=512, elem_id="txt2img_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="الارتفاع", value=512, elem_id="txt2img_height")

                            with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn", tooltip="تبديل العرض/الارتفاع")

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="txt2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='عدد الدُفعات', value=1, elem_id="txt2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='حجم الدفعة', value=1, elem_id="txt2img_batch_size")

                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='مقياس CFG', value=7.0, elem_id="txt2img_cfg_scale")

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        with gr.Row(elem_id="txt2img_accordions", elem_classes="accordions"):
                            with InputAccordion(False, label="إصلاح الدقة العالية", elem_id="txt2img_hr") as enable_hr:
                                with enable_hr.extra():
                                    hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="الدقة بعد التكبير", interactive=False, min_width=0)

                                with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                    hr_upscaler = gr.Dropdown(label="أداة التكبير", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                    hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='عدد خطوات الدقة العالية', value=0, elem_id="txt2img_hires_steps")
                                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='قوة إزالة التشويش', value=0.7, elem_id="txt2img_denoising_strength")

                                with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                    hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="نسبة التكبير", value=2.0, elem_id="txt2img_hr_scale")
                                    hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="تغيير العرض إلى", value=0, elem_id="txt2img_hr_resize_x")
                                    hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="تغيير الارتفاع إلى", value=0, elem_id="txt2img_hr_resize_y")

                                with FormRow(elem_id="txt2img_hires_fix_row3", variant="compact", visible=opts.hires_fix_show_sampler) as hr_sampler_container:

                                    hr_checkpoint_name = gr.Dropdown(label='نقطة الحفظ', elem_id="hr_checkpoint", choices=["استخدام نفس نقطة التحقق"] + modules.sd_models.checkpoint_tiles(use_short=True), value="استخدام نفس نقطة التحقق")
                                    create_refresh_button(hr_checkpoint_name, modules.sd_models.list_models, lambda: {"choices": ["استخدام نفس نقطة التحقق"] + modules.sd_models.checkpoint_tiles(use_short=True)}, "hr_checkpoint_refresh")

                                    hr_sampler_name = gr.Dropdown(label='طريقة أخذ العينات للدقة العالية', elem_id="hr_sampler", choices=["استخدام نفس طريقة أخذ العينات"] + sd_samplers.visible_sampler_names(), value="استخدام نفس طريقة أخذ العينات")
                                    hr_scheduler = gr.Dropdown(label='نوع الجدولة للدقة العالية', elem_id="hr_scheduler", choices=["استخدام نفس نوع الجدولة"] + [x.label for x in sd_schedulers.schedulers], value="استخدام نفس نوع الجدولة")

                                with FormRow(elem_id="txt2img_hires_fix_row4", variant="compact", visible=opts.hires_fix_show_prompts) as hr_prompts_container:
                                    with gr.Column(scale=80):
                                        with gr.Row():
                                            hr_prompt = gr.Textbox(label="نص الدقة العالية", elem_id="hires_prompt", show_label=False, lines=3, placeholder="نص تمريرة إصلاح الدقة العالية.\nاتركه فارغًا لاستخدام نفس النص في التمريرة الأولى.", elem_classes=["prompt"])
                                    with gr.Column(scale=80):
                                        with gr.Row():
                                            hr_negative_prompt = gr.Textbox(label="النص السلبي للدقة العالية", elem_id="hires_neg_prompt", show_label=False, lines=3, placeholder="النص السلبي لتمرير إصلاح الدقة العالية.\nاتركه فارغًا لاستخدام نفس النص السلبي في التمريرة الأولى.", elem_classes=["prompt"])

                            scripts.scripts_txt2img.setup_ui_for_section(category)

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="txt2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='عدد الدُفعات', value=1, elem_id="txt2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='حجم الدفعة', value=1, elem_id="txt2img_batch_size")

                    elif category == "override_settings":
                        with FormRow(elem_id="txt2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('txt2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = scripts.scripts_txt2img.setup_ui()

                    if category not in {"accordions"}:
                        scripts.scripts_txt2img.setup_ui_for_section(category)

            hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]

            for component in hr_resolution_preview_inputs:
                event = component.release if isinstance(component, gr.Slider) else component.change

                event(
                    fn=calc_resolution_hires,
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )
                event(
                    None,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[],
                    show_progress=False,
                )

            output_panel = create_output_panel("txt2img", opts.outdir_txt2img_samples, toprow)

            txt2img_inputs = [
                dummy_component,
                toprow.prompt,
                toprow.negative_prompt,
                toprow.ui_styles.dropdown,
                batch_count,
                batch_size,
                cfg_scale,
                height,
                width,
                enable_hr,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                hr_checkpoint_name,
                hr_sampler_name,
                hr_scheduler,
                hr_prompt,
                hr_negative_prompt,
                override_settings,
            ] + custom_inputs

            txt2img_outputs = [
                output_panel.gallery,
                output_panel.generation_info,
                output_panel.infotext,
                output_panel.html_log,
            ]

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=txt2img_inputs,
                outputs=txt2img_outputs,
                show_progress=False,
            )

            toprow.prompt.submit(**txt2img_args)
            toprow.submit.click(**txt2img_args)

            output_panel.button_upscale.click(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img_upscale, extra_outputs=[None, '', '']),
                _js="submit_txt2img_upscale",
                inputs=txt2img_inputs[0:1] + [output_panel.gallery, dummy_component, output_panel.generation_info] + txt2img_inputs[1:],
                outputs=txt2img_outputs,
                show_progress=False,
            )

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('txt2img')}", inputs=None, outputs=None, show_progress=False)

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                _js="restoreProgressTxt2img",
                inputs=[dummy_component],
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress=False,
            )

            txt2img_paste_fields = [
                PasteField(toprow.prompt, "النص", api="prompt"),
                PasteField(toprow.negative_prompt, "النص السلبي", api="negative_prompt"),
                PasteField(cfg_scale, "مقياس CFG", api="cfg_scale"),
                PasteField(width, "الحجم-1", api="width"),
                PasteField(height, "الحجم-2", api="height"),
                PasteField(batch_size, "حجم الدفعة", api="batch_size"),
                PasteField(toprow.ui_styles.dropdown, lambda d: d["مصفوفة الأنماط"] if isinstance(d.get("مصفوفة الأنماط"), list) else gr.update(), api="styles"),
                PasteField(denoising_strength, "قوة إزالة التشويش", api="denoising_strength"),
                PasteField(enable_hr, lambda d: "قوة إزالة التشويش" in d and ("تكبير الدقة العالية" in d or "أداة تكبير الدقة العالية" in d or "تغيير حجم الدقة العالية-1" in d), api="enable_hr"),
                PasteField(hr_scale, "تكبير الدقة العالية", api="hr_scale"),
                PasteField(hr_upscaler, "أداة تكبير الدقة العالية", api="hr_upscaler"),
                PasteField(hr_second_pass_steps, "خطوات الدقة العالية", api="hr_second_pass_steps"),
                PasteField(hr_resize_x, "تغيير حجم الدقة العالية-1", api="hr_resize_x"),
                PasteField(hr_resize_y, "تغيير حجم الدقة العالية-2", api="hr_resize_y"),
                PasteField(hr_checkpoint_name, "نقطة تحقق الدقة العالية", api="hr_checkpoint_name"),
                PasteField(hr_sampler_name, sd_samplers.get_hr_sampler_from_infotext, api="hr_sampler_name"),
                PasteField(hr_scheduler, sd_samplers.get_hr_scheduler_from_infotext, api="hr_scheduler"),
                PasteField(hr_sampler_container, lambda d: gr.update(visible=True) if d.get("مُولّد الدقة العالية", "استخدام نفس طريقة أخذ العينات") != "استخدام نفس طريقة أخذ العينات" or d.get("نقطة تحقق الدقة العالية", "استخدام نفس نقطة التحقق") != "استخدام نفس نقطة التحقق" or d.get("نوع جدولة الدقة العالية", "استخدام نفس نوع الجدولة") != "استخدام نفس نوع الجدولة" else gr.update()),
                PasteField(hr_prompt, "نص الدقة العالية", api="hr_prompt"),
                PasteField(hr_negative_prompt, "النص السلبي للدقة العالية", api="hr_negative_prompt"),
                PasteField(hr_prompts_container, lambda d: gr.update(visible=True) if d.get("نص الدقة العالية", "") != "" or d.get("النص السلبي للدقة العالية", "") != "" else gr.update()),
                *scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=toprow.paste, tabname="txt2img", source_text_component=toprow.prompt, source_image_component=None,
            ))

            steps = scripts.scripts_txt2img.script('Sampler').steps

            txt2img_preview_params = [
                toprow.prompt,
                toprow.negative_prompt,
                steps,
                scripts.scripts_txt2img.script('Sampler').sampler_name,
                cfg_scale,
                scripts.scripts_txt2img.script('Seed').seed,
                width,
                height,
            ]

            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])
            toprow.token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.negative_token_button.click(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])

        extra_networks_ui = ui_extra_networks.create_ui(txt2img_interface, [txt2img_generation_tab], 'txt2img')
        ui_extra_networks.setup_ui(extra_networks_ui, output_panel.gallery)

        extra_tabs.__exit__()

    scripts.scripts_current = scripts.scripts_img2img
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        toprow = ui_toprow.Toprow(is_img2img=True, is_compact=shared.opts.compact_prompt_box)

        extra_tabs = gr.Tabs(elem_id="img2img_extra_tabs", elem_classes=["extra-networks"])
        extra_tabs.__enter__()

        with gr.Tab("التوليد", id="img2img_generation") as img2img_generation_tab, ResizeHandleRow(equal_height=False):
            with ExitStack() as stack:
                if shared.opts.img2img_settings_accordion:
                    stack.enter_context(gr.Accordion("فتح الإعدادات", open=False))
                stack.enter_context(gr.Column(variant='compact', elem_id="img2img_settings"))

                copy_image_buttons = []
                copy_image_destinations = {}

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}"):
                        gr.HTML("نسخ الصورة إلى: ", elem_id=f"img2img_label_copy_to_{tab_name}")

                        for title, name in zip(['صورة إلى صورة', 'رسم', 'ترميم', 'ترميم مع رسم'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue

                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

                scripts.scripts_img2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    if category == "image":
                        with gr.Tabs(elem_id="mode_img2img"):
                            img2img_selected_tab = gr.Number(value=0, visible=False)

                            with gr.TabItem('img2img', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
                                init_img = gr.Image(label="صورة لـ img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA", height=opts.img2img_editor_height)
                                add_copy_image_controls('img2img', init_img)

                            with gr.TabItem('رسم', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                                sketch = gr.Image(label="صورة لـ img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGB", height=opts.img2img_editor_height, brush_color=opts.img2img_sketch_default_brush_color)
                                add_copy_image_controls('sketch', sketch)

                            with gr.TabItem('ترميم', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                                init_img_with_mask = gr.Image(label="صورة للترميم مع قناع", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=opts.img2img_editor_height, brush_color=opts.img2img_inpaint_mask_brush_color)
                                add_copy_image_controls('inpaint', init_img_with_mask)

                            with gr.TabItem('ترميم مع رسم', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                                inpaint_color_sketch = gr.Image(label="ترميم مع تلوين ورسم", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGB", height=opts.img2img_editor_height, brush_color=opts.img2img_inpaint_sketch_default_brush_color)
                                inpaint_color_sketch_orig = gr.State(None)
                                add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                                def update_orig(image, state):
                                    if image is not None:
                                        same_size = state is not None and state.size == image.size
                                        has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                        edited = same_size and has_exact_match
                                        return image if not edited or state is None else state

                                inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                            with gr.TabItem('ترميم (رفع صورة)', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                                init_img_inpaint = gr.Image(label="صورة لـ img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                                init_mask_inpaint = gr.Image(label="القناع", source="upload", interactive=True, type="pil", image_mode="RGBA", elem_id="img_inpaint_mask")

                            with gr.TabItem('معالجة جماعية', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                                with gr.Tabs(elem_id="img2img_batch_source"):
                                    img2img_batch_source_type = gr.Textbox(visible=False, value="رفع")
                                    with gr.TabItem('رفع ملفات', id='batch_upload', elem_id="img2img_batch_upload_tab") as tab_batch_upload:
                                        img2img_batch_upload = gr.Files(label="الملفات", interactive=True, elem_id="img2img_batch_upload")
                                    with gr.TabItem('من مجلد', id='batch_from_dir', elem_id="img2img_batch_from_dir_tab") as tab_batch_from_dir:
                                        hidden = '<br>معطل عند التشغيل مع --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                                        gr.HTML(
                                            "<p style='padding-bottom: 1em;' class=\"text-gray-500\">معالجة الصور في مجلد موجود على نفس الجهاز الذي يعمل عليه الخادم." +
                                            "<br>استخدم مجلد خرج فارغًا لحفظ الصور في المسار الافتراضي بدل الكتابة في مجلد الخرج." +
                                            f"<br>أضف مجلد أقنعة الترميم للمعالجة الجماعية للترميم."
                                            f"{hidden}</p>"
                                        )
                                        img2img_batch_input_dir = gr.Textbox(label="مجلد الإدخال", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                                        img2img_batch_output_dir = gr.Textbox(label="مجلد الإخراج", **shared.hide_dirs, elem_id="img2img_batch_output_dir")
                                        img2img_batch_inpaint_mask_dir = gr.Textbox(label="مجلد أقنعة الترميم الجماعي (مطلوب فقط للمعالجة الجماعية للترميم)", **shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")
                                tab_batch_upload.select(fn=lambda: "رفع", inputs=[], outputs=[img2img_batch_source_type])
                                tab_batch_from_dir.select(fn=lambda: "من مجلد", inputs=[], outputs=[img2img_batch_source_type])
                                with gr.Accordion("معلومات PNG", open=False):
                                    img2img_batch_use_png_info = gr.Checkbox(label="إضافة معلومات PNG إلى النصوص", elem_id="img2img_batch_use_png_info")
                                    img2img_batch_png_info_dir = gr.Textbox(label="مجلد معلومات PNG", **shared.hide_dirs, placeholder="اتركه فارغًا لاستخدام مجلد الإدخال", elem_id="img2img_batch_png_info_dir")
                                    img2img_batch_png_info_props = gr.CheckboxGroup(["النص", "النص السلبي", "البذرة", "مقياس CFG", "المُولّد", "الخطوات", "هاش النموذج"], label="المعلمات المأخوذة من معلومات PNG", info="سيتم إلحاق النصوص القادمة من معلومات PNG بالنصوص المحددة في الواجهة.")

                            img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]

                            for i, tab in enumerate(img2img_tabs):
                                tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

                        def copy_image(img):
                            if isinstance(img, dict) and 'image' in img:
                                return img['image']

                            return img

                        for button, name, elem in copy_image_buttons:
                            button.click(
                                fn=copy_image,
                                inputs=[elem],
                                outputs=[copy_image_destinations[name]],
                            )
                            button.click(
                                fn=lambda: None,
                                _js=f"switch_to_{name.replace(' ', '_')}",
                                inputs=[],
                                outputs=[],
                            )

                        with FormRow():
                            resize_mode = gr.Radio(label="وضع تغيير الحجم", elem_id="resize_mode", choices=["تغيير الحجم فقط", "قص وتغيير الحجم", "تغيير الحجم وملء", "تغيير الحجم فقط (تكبير كامِن)"], type="index", value="تغيير الحجم فقط")

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                selected_scale_tab = gr.Number(value=0, visible=False)

                                with gr.Tabs(elem_id="img2img_tabs_resize"):
                                    with gr.Tab(label="تغيير الحجم إلى", id="to", elem_id="img2img_tab_resize_to") as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="العرض", value=512, elem_id="img2img_width")
                                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="الارتفاع", value=512, elem_id="img2img_height")
                                            with gr.Column(elem_id="img2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="img2img_res_switch_btn", tooltip="تبديل العرض/الارتفاع")
                                                detect_image_size_btn = ToolButton(value=detect_image_size_symbol, elem_id="img2img_detect_image_size_btn", tooltip="اكتشاف الحجم تلقائياً من img2img")

                                    with gr.Tab(label="تغيير الحجم بنسبة", id="by", elem_id="img2img_tab_resize_by") as tab_scale_by:
                                        scale_by = gr.Slider(minimum=0.05, maximum=4.0, step=0.05, label="مقياس", value=1.0, elem_id="img2img_scale")

                                        with FormRow():
                                            scale_by_html = FormHTML(resize_from_to_html(0, 0, 0.0), elem_id="img2img_scale_resolution_preview")
                                            gr.Slider(label="غير مستخدم", elem_id="img2img_unused_scale_by_slider")
                                            button_update_resize_to = gr.Button(visible=False, elem_id="img2img_update_resize_to")

                                    on_change_args = dict(
                                        fn=resize_from_to_html,
                                        _js="currentImg2imgSourceResolution",
                                        inputs=[dummy_component, dummy_component, scale_by],
                                        outputs=scale_by_html,
                                        show_progress=False,
                                    )

                                    scale_by.release(**on_change_args)
                                    button_update_resize_to.click(**on_change_args)

                            tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
                            tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="img2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='عدد الدُفعات', value=1, elem_id="img2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='حجم الدفعة', value=1, elem_id="img2img_batch_size")

                    elif category == "denoising":
                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='قوة إزالة التشويش', value=0.75, elem_id="img2img_denoising_strength")

                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='مقياس CFG', value=7.0, elem_id="img2img_cfg_scale")
                            image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='مقياس CFG للصورة', value=1.5, elem_id="img2img_image_cfg_scale", visible=False)

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        with gr.Row(elem_id="img2img_accordions", elem_classes="accordions"):
                            scripts.scripts_img2img.setup_ui_for_section(category)

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="img2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='عدد الدُفعات', value=1, elem_id="img2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='حجم الدفعة', value=1, elem_id="img2img_batch_size")

                    elif category == "override_settings":
                        with FormRow(elem_id="img2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('img2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="img2img_script_container"):
                            custom_inputs = scripts.scripts_img2img.setup_ui()

                    elif category == "inpaint":
                        with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                            with FormRow():
                                mask_blur = gr.Slider(label='تشويش القناع', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                                mask_alpha = gr.Slider(label="شفافية القناع", visible=False, elem_id="img2img_mask_alpha")

                            with FormRow():
                                inpainting_mask_invert = gr.Radio(label='وضع القناع', choices=['ترميم المنطقة المغطاة', 'ترميم المنطقة غير المغطاة'], value='ترميم المنطقة المغطاة', type="index", elem_id="img2img_mask_mode")

                            with FormRow():
                                inpainting_fill = gr.Radio(label='محتوى المنطقة المقنَّعة', choices=['ملء', 'أصلي', 'ضجيج كامِن', 'بدون كامِن'], value='أصلي', type="index", elem_id="img2img_inpainting_fill")

                            with FormRow():
                                with gr.Column():
                                    inpaint_full_res = gr.Radio(label="منطقة الترميم", choices=["الصورة كاملة", "المنطقة المقنَّعة فقط"], type="index", value="الصورة كاملة", elem_id="img2img_inpaint_full_res")

                                with gr.Column(scale=4):
                                    inpaint_full_res_padding = gr.Slider(label='هامش المنطقة المقنَّعة (بالبكسل)', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                    if category not in {"accordions"}:
                        scripts.scripts_img2img.setup_ui_for_section(category)

            # the code below is meant to update the resolution label after the image in the image selection UI has changed.
            # as it is now the event keeps firing continuously for inpaint edits, which ruins the page with constant requests.
            # I assume this must be a gradio bug and for now we'll just do it for non-inpaint inputs.
            for component in [init_img, sketch]:
                component.change(fn=lambda: None, _js="updateImg2imgResizeToTextAfterChangingImage", inputs=[], outputs=[], show_progress=False)

            def select_img2img_tab(tab):
                return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3),

            for i, elem in enumerate(img2img_tabs):
                elem.select(
                    fn=lambda tab=i: select_img2img_tab(tab),
                    inputs=[],
                    outputs=[inpaint_controls, mask_alpha],
                )

            output_panel = create_output_panel("img2img", opts.outdir_img2img_samples, toprow)

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
                _js="submit_img2img",
                inputs=[
                    dummy_component,
                    dummy_component,
                    toprow.prompt,
                    toprow.negative_prompt,
                    toprow.ui_styles.dropdown,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    inpaint_color_sketch_orig,
                    init_img_inpaint,
                    init_mask_inpaint,
                    mask_blur,
                    mask_alpha,
                    inpainting_fill,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    image_cfg_scale,
                    denoising_strength,
                    selected_scale_tab,
                    height,
                    width,
                    scale_by,
                    resize_mode,
                    inpaint_full_res,
                    inpaint_full_res_padding,
                    inpainting_mask_invert,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    img2img_batch_inpaint_mask_dir,
                    override_settings,
                    img2img_batch_use_png_info,
                    img2img_batch_png_info_props,
                    img2img_batch_png_info_dir,
                    img2img_batch_source_type,
                    img2img_batch_upload,
                ] + custom_inputs,
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress=False,
            )

            interrogate_args = dict(
                _js="get_img2img_tab_index",
                inputs=[
                    dummy_component,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    init_img_inpaint,
                ],
                outputs=[toprow.prompt, dummy_component],
            )

            toprow.prompt.submit(**img2img_args)
            toprow.submit.click(**img2img_args)

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('img2img')}", inputs=None, outputs=None, show_progress=False)

            detect_image_size_btn.click(
                fn=lambda w, h, _: (w or gr.update(), h or gr.update()),
                _js="currentImg2imgSourceResolution",
                inputs=[dummy_component, dummy_component, dummy_component],
                outputs=[width, height],
                show_progress=False,
            )

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                _js="restoreProgressImg2img",
                inputs=[dummy_component],
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress=False,
            )

            toprow.button_interrogate.click(
                fn=lambda *args: process_interrogate(interrogate, *args),
                **interrogate_args,
            )

            toprow.button_deepbooru.click(
                fn=lambda *args: process_interrogate(interrogate_deepbooru, *args),
                **interrogate_args,
            )

            steps = scripts.scripts_img2img.script('Sampler').steps

            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])
            toprow.token_button.click(fn=update_token_counter, inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.negative_token_button.click(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])

            img2img_paste_fields = [
                (toprow.prompt, "النص"),
                (toprow.negative_prompt, "النص السلبي"),
                (cfg_scale, "مقياس CFG"),
                (image_cfg_scale, "مقياس CFG للصورة"),
                (width, "الحجم-1"),
                (height, "الحجم-2"),
                (batch_size, "حجم الدفعة"),
                (toprow.ui_styles.dropdown, lambda d: d["مصفوفة الأنماط"] if isinstance(d.get("مصفوفة الأنماط"), list) else gr.update()),
                (denoising_strength, "قوة إزالة التشويش"),
                (mask_blur, "تشويش القناع"),
                (inpainting_mask_invert, 'وضع القناع'),
                (inpainting_fill, 'محتوى المنطقة المقنَّعة'),
                (inpaint_full_res, 'منطقة الترميم'),
                (inpaint_full_res_padding, 'هامش المنطقة المقنَّعة'),
                *scripts.scripts_img2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields, override_settings)
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=toprow.paste, tabname="img2img", source_text_component=toprow.prompt, source_image_component=None,
            ))

        extra_networks_ui_img2img = ui_extra_networks.create_ui(img2img_interface, [img2img_generation_tab], 'img2img')
        ui_extra_networks.setup_ui(extra_networks_ui_img2img, output_panel.gallery)

        extra_tabs.__exit__()

    scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        ui_postprocessing.create_ui()

    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        with ResizeHandleRow(equal_height=False):
            with gr.Column(variant='panel'):
                image = gr.Image(elem_id="pnginfo_image", label="المصدر", source="upload", interactive=True, type="pil")

            with gr.Column(variant='panel'):
                html = gr.HTML()
                generation_info = gr.Textbox(visible=False, elem_id="pnginfo_generation_info")
                html2 = gr.HTML()
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=generation_info, source_image_component=image,
                    ))

        image.change(
            fn=wrap_gradio_call_no_job(modules.extras.run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )

    modelmerger_ui = ui_checkpoint_merger.UiCheckpointMerger()

    with gr.Blocks(analytics_enabled=False) as train_interface:
        with gr.Row(equal_height=False):
            gr.HTML(value="<p style='margin-bottom: 0.7em'>لمزيد من الشرح التفصيلي، راجع صفحة الـ <b><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\">ويكي</a></b>.</p>")

        with ResizeHandleRow(variant="compact", equal_height=False):
            with gr.Tabs(elem_id="train_tabs"):

                with gr.Tab(label="إنشاء تضمين", id="create_embedding"):
                    new_embedding_name = gr.Textbox(label="الاسم", elem_id="train_new_embedding_name")
                    initialization_text = gr.Textbox(label="نص التهيئة", value="*", elem_id="train_initialization_text")
                    nvpt = gr.Slider(label="عدد المتجهات لكل رمز", minimum=1, maximum=75, step=1, value=1, elem_id="train_nvpt")
                    overwrite_old_embedding = gr.Checkbox(value=False, label="الكتابة فوق التضمين القديم", elem_id="train_overwrite_old_embedding")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_embedding = gr.Button(value="إنشاء تضمين", variant='primary', elem_id="train_create_embedding")

                with gr.Tab(label="إنشاء Hypernetwork", id="create_hypernetwork"):
                    new_hypernetwork_name = gr.Textbox(label="الاسم", elem_id="train_new_hypernetwork_name")
                    new_hypernetwork_sizes = gr.CheckboxGroup(label="الوحدات", value=["768", "320", "640", "1280"], choices=["768", "1024", "320", "640", "1280"], elem_id="train_new_hypernetwork_sizes")
                    new_hypernetwork_layer_structure = gr.Textbox("1, 2, 1", label="أدخل بنية طبقات الـ Hypernetwork", placeholder="يجب أن يكون أول وآخر رقم 1. مثال: '1, 2, 1'", elem_id="train_new_hypernetwork_layer_structure")
                    new_hypernetwork_activation_func = gr.Dropdown(value="linear", label="اختر دالة التفعيل للـ Hypernetwork. يُنصح بـ: Swish / Linear(none)", choices=hypernetworks_ui.keys, elem_id="train_new_hypernetwork_activation_func")
                    new_hypernetwork_initialization_option = gr.Dropdown(value = "Normal", label="اختر طريقة تهيئة أوزان الطبقات. يُنصح بـ: Kaiming لـ relu، وXavier لـ sigmoid، وNormal للباقي", choices=["Normal", "KaimingUniform", "KaimingNormal", "XavierUniform", "XavierNormal"], elem_id="train_new_hypernetwork_initialization_option")
                    new_hypernetwork_add_layer_norm = gr.Checkbox(label="إضافة تسوية للطبقات (Layer Norm)", elem_id="train_new_hypernetwork_add_layer_norm")
                    new_hypernetwork_use_dropout = gr.Checkbox(label="استخدام Dropout", elem_id="train_new_hypernetwork_use_dropout")
                    new_hypernetwork_dropout_structure = gr.Textbox("0, 0, 0", label="أدخل بنية الـ Dropout في الـ Hypernetwork (أو اتركه فارغًا). يُنصح بقيم بين 0 و 0.35 مثل: 0, 0.05, 0.15", placeholder="يجب أن يكون أول وآخر رقم 0 والقيم بين 0 و 1. مثال: '0, 0.01, 0'")
                    overwrite_old_hypernetwork = gr.Checkbox(value=False, label="الكتابة فوق الـ Hypernetwork القديم", elem_id="train_overwrite_old_hypernetwork")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_hypernetwork = gr.Button(value="إنشاء Hypernetwork", variant='primary', elem_id="train_create_hypernetwork")

                def get_textual_inversion_template_names():
                    return sorted(textual_inversion.textual_inversion_templates)

                with gr.Tab(label="التدريب", id="train"):
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>قم بتدريب تضمين أو Hypernetwork؛ يجب أن تحدد مجلدًا يحتوي على صور بنسبة أبعاد 1:1 <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[ويكي]</a></p>")
                    with FormRow():
                        train_embedding_name = gr.Dropdown(label='التضمين', elem_id="train_embedding", choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
                        create_refresh_button(train_embedding_name, sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings, lambda: {"choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())}, "refresh_train_embedding_name")

                        train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork", choices=sorted(shared.hypernetworks))
                        create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks, lambda: {"choices": sorted(shared.hypernetworks)}, "refresh_train_hypernetwork_name")

                    with FormRow():
                        embedding_learn_rate = gr.Textbox(label='معدل تعلم التضمين', placeholder="معدل تعلم التضمين", value="0.005", elem_id="train_embedding_learn_rate")
                        hypernetwork_learn_rate = gr.Textbox(label='معدل تعلم الـ Hypernetwork', placeholder="معدل تعلم الـ Hypernetwork", value="0.00001", elem_id="train_hypernetwork_learn_rate")

                    with FormRow():
                        clip_grad_mode = gr.Dropdown(value="معطل", label="قصّ التدرجات (Gradient Clipping)", choices=["معطل", "قيمة", "معياري"])
                        clip_grad_value = gr.Textbox(placeholder="قيمة قصّ التدرج", value="0.1", show_label=False)

                    with FormRow():
                        batch_size = gr.Number(label='حجم الدفعة', value=1, precision=0, elem_id="train_batch_size")
                        gradient_step = gr.Number(label='خطوات تجميع التدرجات', value=1, precision=0, elem_id="train_gradient_step")

                    dataset_directory = gr.Textbox(label='مجلد بيانات التدريب', placeholder="مسار المجلد الذي يحتوي على الصور", elem_id="train_dataset_directory")
                    log_directory = gr.Textbox(label='مجلد السجلات', placeholder="مسار المجلد الذي سيتم حفظ النتائج فيه", value="textual_inversion", elem_id="train_log_directory")

                    with FormRow():
                        template_file = gr.Dropdown(label='قالب النص', value="style_filewords.txt", elem_id="train_template_file", choices=get_textual_inversion_template_names())
                        create_refresh_button(template_file, textual_inversion.list_textual_inversion_templates, lambda: {"choices": get_textual_inversion_template_names()}, "refrsh_train_template_file")

                    training_width = gr.Slider(minimum=64, maximum=2048, step=8, label="العرض", value=512, elem_id="train_training_width")
                    training_height = gr.Slider(minimum=64, maximum=2048, step=8, label="الارتفاع", value=512, elem_id="train_training_height")
                    varsize = gr.Checkbox(label="عدم تغيير حجم الصور", value=False, elem_id="train_varsize")
                    steps = gr.Number(label='أقصى عدد خطوات', value=100000, precision=0, elem_id="train_steps")

                    with FormRow():
                        create_image_every = gr.Number(label='حفظ صورة في مجلد السجلات كل N خطوة، 0 للتعطيل', value=500, precision=0, elem_id="train_create_image_every")
                        save_embedding_every = gr.Number(label='حفظ نسخة من التضمين في مجلد السجلات كل N خطوة، 0 للتعطيل', value=500, precision=0, elem_id="train_save_embedding_every")

                    use_weight = gr.Checkbox(label="استخدام قناة ألفا في PNG كوزن للخسارة", value=False, elem_id="use_weight")

                    save_image_with_stored_embedding = gr.Checkbox(label='حفظ الصور مع التضمين داخل بيانات PNG', value=True, elem_id="train_save_image_with_stored_embedding")
                    preview_from_txt2img = gr.Checkbox(label='قراءة الإعدادات (النص، إلخ) من تبويب txt2img عند إنشاء المعاينات', value=False, elem_id="train_preview_from_txt2img")

                    shuffle_tags = gr.Checkbox(label="خلط الوسوم المفصولة بـ ',' عند إنشاء النصوص.", value=False, elem_id="train_shuffle_tags")
                    tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="إسقاط بعض الوسوم عشوائيًا عند إنشاء النصوص.", value=0, elem_id="train_tag_drop_out")

                    latent_sampling_method = gr.Radio(label='اختر طريقة أخذ العينات في الفضاء الكامن', value="مرة واحدة", choices=['مرة واحدة', 'حتمي', 'عشوائي'], elem_id="train_latent_sampling_method")

                    with gr.Row():
                        train_embedding = gr.Button(value="تدريب التضمين", variant='primary', elem_id="train_train_embedding")
                        interrupt_training = gr.Button(value="إيقاف", elem_id="train_interrupt_training")
                        train_hypernetwork = gr.Button(value="تدريب الـ Hypernetwork", variant='primary', elem_id="train_train_hypernetwork")

                params = script_callbacks.UiTrainTabParams(txt2img_preview_params)

                script_callbacks.ui_train_tabs_callback(params)

            with gr.Column(elem_id='ti_gallery_container'):
                ti_output = gr.Text(elem_id="ti_output", value="", show_label=False)
                gr.Gallery(label='النتيجة', show_label=False, elem_id='ti_gallery', columns=4)
                gr.HTML(elem_id="ti_progress", value="")
                ti_outcome = gr.HTML(elem_id="ti_error", value="")

        create_embedding.click(
            fn=textual_inversion_ui.create_embedding,
            inputs=[
                new_embedding_name,
                initialization_text,
                nvpt,
                overwrite_old_embedding,
            ],
            outputs=[
                train_embedding_name,
                ti_output,
                ti_outcome,
            ]
        )

        create_hypernetwork.click(
            fn=hypernetworks_ui.create_hypernetwork,
            inputs=[
                new_hypernetwork_name,
                new_hypernetwork_sizes,
                overwrite_old_hypernetwork,
                new_hypernetwork_layer_structure,
                new_hypernetwork_activation_func,
                new_hypernetwork_initialization_option,
                new_hypernetwork_add_layer_norm,
                new_hypernetwork_use_dropout,
                new_hypernetwork_dropout_structure
            ],
            outputs=[
                train_hypernetwork_name,
                ti_output,
                ti_outcome,
            ]
        )

        train_embedding.click(
            fn=wrap_gradio_gpu_call(textual_inversion_ui.train_embedding, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_embedding_name,
                embedding_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        train_hypernetwork.click(
            fn=wrap_gradio_gpu_call(hypernetworks_ui.train_hypernetwork, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_hypernetwork_name,
                hypernetwork_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
    ui_settings_from_file = loadsave.ui_settings.copy()

    settings.create_ui(loadsave, dummy_component)

    interfaces = [
        (txt2img_interface, "تحويل النص الى صورة", "تحويل النص الى صورة"),
        (img2img_interface, "تحويل صورة الى صورة", "img2img"),
        (extras_interface, "أدوات إضافية", "extras"),
        (pnginfo_interface, "معلومات PNG", "pnginfo"),
        (modelmerger_ui.blocks, "دمج نقاط الحفظ", "modelmerger"),
        (train_interface, "التدريب", "train"),
    ]

    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings.interface, "الإعدادات", "settings")]

    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "الإضافات", "extensions")]

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    with gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="توليد الصور بالذكاء الاصطناعي") as demo:
        settings.add_quicksettings()

        parameters_copypaste.connect_paste_params_buttons()
       
        with gr.Tabs(elem_id="tabs") as tabs:
            tab_order = {k: i for i, k in enumerate(opts.ui_tab_order)}
            sorted_interfaces = sorted(interfaces, key=lambda x: tab_order.get(x[1], 9999))

            for interface, label, ifid in sorted_interfaces:
                if label in shared.opts.hidden_tabs:
                    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    interface.render()

                if ifid not in ["extensions", "settings"]:
                    loadsave.add_block(interface, ifid)

            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)

            loadsave.setup_ui()

        if os.path.exists(os.path.join(script_path, "notification.mp3")) and shared.opts.notification_audio:
            gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        footer = shared.html("footer.html")
        footer = footer.format(versions=versions_html(), api_docs="/docs" if shared.cmd_opts.api else "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API")
        gr.HTML(footer, elem_id="footer")

        settings.add_functionality(demo)

        update_image_cfg_scale_visibility = lambda: gr.update(visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit")
        settings.text_settings.change(fn=update_image_cfg_scale_visibility, inputs=[], outputs=[image_cfg_scale])
        demo.load(fn=update_image_cfg_scale_visibility, inputs=[], outputs=[image_cfg_scale])

        modelmerger_ui.setup_ui(dummy_component=dummy_component, sd_model_checkpoint_component=settings.component_dict['sd_model_checkpoint'])

    if ui_settings_from_file != loadsave.ui_settings:
        loadsave.dump_defaults()
    demo.ui_loadsave = loadsave

    return demo


def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    tag = launch.git_tag()

    if shared.xformers_available:
        import xformers
        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    return f"""
version: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{tag}</a>
&#x2000;•&#x2000;
python: <span title="{sys.version}">{python_version}</span>
&#x2000;•&#x2000;
torch: {getattr(torch, '__long_version__',torch.__version__)}
&#x2000;•&#x2000;
xformers: {xformers_version}
&#x2000;•&#x2000;
gradio: {gr.__version__}
&#x2000;•&#x2000;
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
"""


def setup_ui_api(app):
    from pydantic import BaseModel, Field

    class QuicksettingsHint(BaseModel):
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=list[QuicksettingsHint])

    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])

    app.add_api_route("/internal/profile-startup", lambda: timer.startup_record, methods=["GET"])

    def download_sysinfo(attachment=False):
        from fastapi.responses import PlainTextResponse

        text = sysinfo.get()
        filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"

        return PlainTextResponse(text, headers={'Content-Disposition': f'{"attachment" if attachment else "inline"}; filename="{filename}"'})

    app.add_api_route("/internal/sysinfo", download_sysinfo, methods=["GET"])
    app.add_api_route("/internal/sysinfo-download", lambda: download_sysinfo(attachment=True), methods=["GET"])

    import fastapi.staticfiles
    app.mount("/webui-assets", fastapi.staticfiles.StaticFiles(directory=launch_utils.repo_dir('stable-diffusion-webui-assets')), name="webui-assets")
