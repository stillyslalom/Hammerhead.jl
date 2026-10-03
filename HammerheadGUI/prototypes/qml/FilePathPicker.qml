import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

RowLayout {
    id: picker
    property alias text: pathField.text
    property alias fieldText: pathField.text
    property alias placeholderText: pathField.placeholderText
    property var dialog: null
    readonly property bool dialogVisible: dialog !== null && dialog.visible
    property string purpose: ""
    property string fieldName: ""
    property string dialogTitle: "Choose a file"
    property bool saveMode: false
    property bool dialogsEnabled: true
    property bool forceNonNative: false
    property int dialogPopupType: Popup.Window
    property int additionalDialogOptions: 0
    property int token: 0
    // Allows the hidden harness to fail the platform opening step after the
    // token is acquired, rather than bypassing the lifecycle being tested.
    property var openChooser: function(chosen) { chosen.open(); }
    signal submitted(string path)
    signal draftAccepted(string path)
    Layout.fillWidth: true
    spacing: 4

    function openDialog() {
        if (!dialogsEnabled || token !== 0) return false;
        var next = Julia.begin_file_dialog(purpose);
        if (next === 0) return false;
        token = next;
        try {
            var chosen = dialogComponent.createObject(picker, { requestToken: next, requestPurpose: purpose });
            if (chosen === null) throw new Error("file dialog could not be created");
            dialog = chosen;
            chosen.currentFolder = Julia.file_dialog_folder(pathField.text);
            openChooser(chosen);
            return true;
        } catch (error) {
            Julia.reject_file_dialog(token);
            token = 0;
            if (dialog !== null) finishChooser(dialog);
            throw error;
        }
    }
    function dismiss() {
        if (token !== 0) Julia.reject_file_dialog(token);
        token = 0;
        if (dialog !== null) finishChooser(dialog);
    }
    function finishChooser(chosen) {
        chosen.finished = true;
        if (dialog === chosen) { dialog = null; token = 0; }
        chosen.close();
        chosen.destroy();
    }
    TextField {
        id: pathField
        objectName: picker.fieldName
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        Accessible.name: picker.dialogTitle
        onAccepted: picker.submitted(text)
    }
    Button {
        objectName: picker.fieldName + "Browse"
        text: "Browse"
        enabled: picker.dialogsEnabled && picker.token === 0
        Accessible.name: "Browse: " + picker.dialogTitle
        onClicked: picker.openDialog()
    }
    Component {
      id: dialogComponent
      FileDialog {
        id: chooser
        property int requestToken: 0
        property string requestPurpose: ""
        property bool finished: false
        objectName: picker.fieldName + "Dialog"
        title: picker.dialogTitle
        acceptLabel: picker.saveMode ? "Use path" : "Choose file"
        fileMode: picker.saveMode ? FileDialog.SaveFile : FileDialog.OpenFile
        nameFilters: ["Hammerhead files (*.jld2)", "All files (*)"]
        defaultSuffix: picker.saveMode ? "jld2" : ""
        popupType: picker.dialogPopupType
        options: (picker.forceNonNative ? FileDialog.DontUseNativeDialog : 0) |
                 (picker.saveMode ? FileDialog.DontConfirmOverwrite : 0) | picker.additionalDialogOptions
        onAccepted: {
            if (finished) return;
            finished = true;
            try {
                var path = Julia.accept_file_dialog(requestToken, requestPurpose, selectedFile);
                if (path.length > 0) {
                    pathField.text = path;
                    picker.draftAccepted(path);
                }
            } finally {
                picker.finishChooser(chooser);
            }
        }
        onRejected: {
            if (finished) return;
            finished = true;
            try { Julia.reject_file_dialog(requestToken); }
            finally { picker.finishChooser(chooser); }
        }
        Component.onDestruction: { if (!finished) Julia.reject_file_dialog(requestToken); }
      }
    }
    Component.onDestruction: picker.dismiss()
}
