
function extensions_apply(_disabled_list, _update_list, disable_all) {
    var disable = [];
    var update = [];
    const extensions_input = gradioApp().querySelectorAll('#extensions input[type="checkbox"]');
    if (extensions_input.length == 0) {
        throw Error("صفحة الإضافات لم يتم تحميلها بعد.");
    }
    extensions_input.forEach(function(x) {
        if (x.name.startsWith("enable_") && !x.checked) {
            disable.push(x.name.substring(7));
        }

        if (x.name.startsWith("update_") && x.checked) {
            update.push(x.name.substring(7));
        }
    });

    restart_reload();

    return [JSON.stringify(disable), JSON.stringify(update), disable_all];
}

function extensions_check() {
    var disable = [];

    gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach(function(x) {
        if (x.name.startsWith("enable_") && !x.checked) {
            disable.push(x.name.substring(7));
        }
    });

    gradioApp().querySelectorAll('#extensions .extension_status').forEach(function(x) {
            x.innerHTML = "جاري التحميل...";
    });


    var id = randomId();
    requestProgress(id, gradioApp().getElementById('extensions_installed_html'), null, function() {

    });

    return [id, JSON.stringify(disable)];
}

function install_extension_from_index(button, url) {
    button.disabled = "disabled";
    button.value = "جاري التثبيت...";

    var textarea = gradioApp().querySelector('#extension_to_install textarea');
    textarea.value = url;
    updateInput(textarea);

    gradioApp().querySelector('#install_extension_button').click();
}

function config_state_confirm_restore(_, config_state_name, config_restore_type) {
    if (config_state_name == "الحالي") {
        return [false, config_state_name, config_restore_type];
    }
    let restored = "";
    // تحويل القيم العربية إلى إنجليزية للتحقق
    if (config_restore_type == "الإضافات فقط" || config_restore_type == "extensions") {
        restored = "جميع إصدارات الإضافات المحفوظة";
    } else if (config_restore_type == "واجهة الويب فقط" || config_restore_type == "webui") {
        restored = "إصدار واجهة الويب";
    } else if (config_restore_type == "كلاهما" || config_restore_type == "both") {
        restored = "إصدار واجهة الويب وجميع إصدارات الإضافات المحفوظة";
    }
    let confirmed = confirm("هل أنت متأكد أنك تريد الاستعادة من هذه الحالة؟\nسيتم إعادة تعيين " + restored + ".");
    if (confirmed) {
        restart_reload();
        gradioApp().querySelectorAll('#extensions .extension_status').forEach(function(x) {
            x.innerHTML = "جاري التحميل...";
        });
    }
    return [confirmed, config_state_name, config_restore_type];
}

function toggle_all_extensions(event) {
    gradioApp().querySelectorAll('#extensions .extension_toggle').forEach(function(checkbox_el) {
        checkbox_el.checked = event.target.checked;
    });
}

function toggle_extension() {
    let all_extensions_toggled = true;
    for (const checkbox_el of gradioApp().querySelectorAll('#extensions .extension_toggle')) {
        if (!checkbox_el.checked) {
            all_extensions_toggled = false;
            break;
        }
    }

    gradioApp().querySelector('#extensions .all_extensions_toggle').checked = all_extensions_toggled;
}
