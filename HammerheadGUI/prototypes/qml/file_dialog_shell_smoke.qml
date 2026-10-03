// Opt-in automation only: real Quick FileDialogs and synthetic QtTest keys.
import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtTest
import jlqml

Item {
    id: test
    property var shell: null
    property string issued: ""
    property string pending: ""
    property var picker: null
    property string beforeText: ""
    property int beforeToken: 0
    property int waited: 0
    property bool beforeViewport: true
    property int selectedIndex: -1
    TestCase { id: keys; name: "OffscreenDialogKeys"; when: false }

    function visualFind(item, name) {
        if (!item) return null;
        if (item.visible === false) return null;
        if (item.objectName === name) return item;
        if (!item.children) return null;
        for (var i = 0; i < item.children.length; ++i) {
            var found = visualFind(item.children[i], name);
            if (found) return found;
        }
        return null;
    }
    function dialogControl(name) {
        return visualFind(picker.Overlay.overlay, name);
    }

    function choose(purpose) {
        shell.analysisLaneControl.currentIndex = purpose === "result" ? 0 : 1;
        shell.savedSectionControl.currentIndex = 0;
        if (purpose === "record") return shell.savedRecordPicker;
        if (purpose === "result") return shell.nativeResultPicker;
        if (purpose === "output") return shell.replayOutputPicker;
        return shell.runHistoryPicker;
    }
    function startDialog(purpose) {
        picker = choose(purpose);
        beforeText = picker.fieldText;
        beforeViewport = shell.viewportOpen;
        picker.dialogPopupType = Popup.Item;
        if (!picker.forceNonNative)
            throw new Error("automated dialog did not force DontUseNativeDialog");
        if (!picker.openDialog()) throw new Error("picker refused the idle smoke request");
        if (!(picker.dialog.options & FileDialog.DontUseNativeDialog)) throw new Error("actual chooser used native options");
        if (!!(picker.dialog.options & FileDialog.DontConfirmOverwrite) !== picker.saveMode)
            throw new Error("production overwrite option does not match draft-only SaveFile mode");
        beforeToken = picker.token;
    }
    function acknowledge() {
        var purpose = issued.split("-").pop();
        Julia.file_dialog_probe(issued, purpose, picker ? picker.token : 0,
            picker ? picker.dialogVisible : false, picker ? picker.fieldText : "");
        pending = "";
    }
    function browseFits() {
        // Visible field and Browse-button bounds, not merely a model width.
        var purposes = ["result", "record", "output", "history"];
        for (var i = 0; i < purposes.length; ++i) {
            var p = choose(purposes[i]);
            if (!(p.width > 120)) return false;
            var field = null, button = null, fields = 0, buttons = 0;
            for (var j = 0; j < p.children.length; ++j) {
                var c = p.children[j];
                if (c.objectName === p.fieldName) { field = c; fields += 1; }
                if (c.objectName === p.fieldName + "Browse") { button = c; buttons += 1; }
                if (c.objectName === p.fieldName || c.objectName === p.fieldName + "Browse") {
                    var pos = c.mapToItem(shell.contentItem, 0, 0);
                    if (!c.visible || c.width <= 0 || pos.x < 0 || pos.x + c.width > shell.width || pos.y < 0 || pos.y + c.height > shell.height)
                        return false;
                }
            }
            if (fields !== 1 || buttons !== 1 || field.width <= 0 || field.x + field.width > button.x) return false;
        }
        shell.analysisLaneControl.currentIndex = 1;
        return true;
    }
    function capture(label) {
        if (!browseFits()) throw new Error("compact Browse controls are unreachable");
        if (!shell.captureSurface.grabToImage(function(result) {
            if (!result.saveToFile(fileDialogCapturePrefix + "-" + label + ".png")) {
                Julia.smoke_failed("file-dialog capture failed"); return;
            }
            Julia.file_dialog_probe("capture-" + label, "", 0, false,
                shell.width + "," + shell.height + ",1");
            pending = "";
        })) throw new Error("shell capture was rejected");
    }
    function issue(command) {
        issued = command; pending = command; waited = 0;
        var action = command.split("-")[0];
        var purpose = command.split("-").pop();
        if (action === "open") startDialog(purpose);
        else if (action === "select") {
            selectedIndex = -1;
            beforeText = picker.fieldText;
            var encoded = String(shell.uiModel.fileDialogSmokeUrl);
            picker.dialog.currentFolder = encoded.substring(0, encoded.lastIndexOf("/"));
            if (picker.fieldText !== beforeText) throw new Error("selection changed draft before acceptance");
        } else if (action === "accept") picker.dialog.accept();
        else if (action === "reject" || action === "escape" || action === "modal" || action === "stale" || action === "busy") startDialog(purpose);
        else if (action === "opening") {
            picker = choose(purpose);
            var original = picker.openChooser;
            picker.openChooser = function() { throw new Error("injected chooser opening failure"); };
            var threw = false;
            try { picker.openDialog(); } catch (error) { threw = true; }
            picker.openChooser = original;
            if (!threw || picker.token !== 0 || shell.uiModel.fileDialogOpen) throw new Error("opening failure retained its busy token");
            if (!picker.openDialog()) throw new Error("opening failure prevented the next request");
            picker.dialog.reject();
        } else if (action === "capture") {
            shell.width = purpose === "small" ? 900 : 1100;
            shell.height = purpose === "small" ? 600 : 800;
            shell.analysisLaneControl.currentIndex = 1; shell.savedSectionControl.currentIndex = 0;
        } else if (action === "shutdown") startDialog(purpose);
        else throw new Error("unknown dialog command " + command);
    }
    function advance() {
        var action = issued.split("-")[0];
        var purpose = issued.split("-").pop();
        if (action === "open") {
            if (picker.dialogVisible) {
                if (issued === "open-record") {
                    if (waited < 7) return; // settle the asynchronous folder list
                    pending = "capture-open-wait";
                    // QtTest snapshots the window's content and Popup.Item
                    // overlay without Item.grabToImage's QML-engine restriction.
                    var image = keys.grabImage(shell.contentItem);
                    if (!(image.width > 0 && image.height > 0)) throw new Error("actual chooser snapshot was empty");
                    image.save(fileDialogCapturePrefix + "-dialog.png");
                    Julia.file_dialog_probe("dialog-capture", "record", picker.token, true, shell.width + "," + shell.height + ",1");
                    acknowledge();
                } else acknowledge();
            }
        } else if (action === "select") {
            if (picker.fieldText !== beforeText) throw new Error("selection changed the draft");
            if (waited < 4) return;
            // Automation of the installed Quick fallback, not native OS UI.
            // Set the actual save-name control or click a listed existing file.
            if (picker.saveMode) {
                var nameField = dialogControl("fileNameTextField");
                if (!nameField || !nameField.visible) return;
                var savePath = String(shell.uiModel.fileDialogSmokePath).replace(/\\/g, "/");
                nameField.text = savePath.substring(savePath.lastIndexOf("/") + 1);
                nameField.editingFinished();
                console.log("Dialog smoke save-name", nameField.text, picker.dialog.selectedFile);
            } else {
                var list = dialogControl("fileDialogListView");
                if (!list) return;
                var path = String(shell.uiModel.fileDialogSmokePath).replace(/\\/g, "/");
                var name = path.substring(path.lastIndexOf("/") + 1);
                var index = -1;
                for (var row = 0; row < list.count; ++row)
                    if (list.model.get(row, "fileName") === name) { index = row; break; }
                if (index < 0) return;
                if (selectedIndex !== index) {
                    selectedIndex = index;
                    list.positionViewAtIndex(index, ListView.Contain);
                    return; // let the live view lay out the actual click target
                }
                var delegate = list.itemAtIndex(index);
                if (!delegate || !delegate.visible || delegate.width <= 0 || delegate.height <= 0) return;
                console.log("Dialog smoke actual click", issued, list.objectName, list.visible,
                    delegate.objectName, delegate.visible, delegate.width, delegate.height, picker.dialog.currentFolder);
                keys.mouseClick(delegate);
            }
            acknowledge();
        } else if (action === "accept" || action === "opening") {
            if (!picker.dialogVisible) acknowledge();
        } else if (action === "capture") {
            if (waited > 3) { pending = "capture-wait"; capture(purpose); }
        } else if (action === "shutdown") {
            if (!picker.dialogVisible) return;
            shell.close();
            Julia.file_dialog_probe(issued, purpose, picker.token, picker.dialogVisible, picker.fieldText);
            pending = "";
        } else {
            if (!picker.dialogVisible) return;
            if (action !== "modal") picker.dialog.selectedFile = shell.uiModel.fileDialogSmokeUrl;
            if (action === "reject") picker.dialog.reject();
            else if (action === "escape") keys.keyClick(Qt.Key_Escape);
            else if (action === "modal") {
                var key = issued.split("-")[1];
                if (key === "left") keys.keyClick(Qt.Key_Left);
                else if (key === "right") keys.keyClick(Qt.Key_Right);
                else if (key === "close") keys.keyClick(Qt.Key_W, Qt.ControlModifier);
                else keys.keyClick(Qt.Key_Return, Qt.ControlModifier);
                if (shell.viewportOpen !== beforeViewport) throw new Error("modal key closed the viewport behind the dialog");
                // Return may accept the dialog itself. Never send Escape after
                // it closes, when that key would legitimately reach the root.
                if (picker.dialogVisible) keys.keyClick(Qt.Key_Escape);
            } else if (action === "stale") {
                var old = picker.dialog;
                var oldToken = picker.token;
                old.reject();
                if (!picker.openDialog()) throw new Error("same picker could not reopen");
                var fresh = picker.dialog;
                var freshToken = picker.token;
                old.accepted(); old.rejected();
                if (picker.dialog !== fresh || picker.token !== freshToken || freshToken <= oldToken || !shell.uiModel.fileDialogOpen || picker.fieldText !== beforeText)
                    throw new Error("old chooser callback cleared or accepted the new request");
                fresh.reject();
            } else if (action === "busy") {
                Julia.file_dialog_probe("busy-prepare", purpose, picker.token, true, picker.fieldText);
                picker.dialog.accept();
            }
            if (picker.fieldText !== beforeText && issued !== "modal-run-output") throw new Error("rejected/stale/busy dialog changed the draft");
            pending = "closed-wait";
        }
    }
    Timer {
        interval: 40; repeat: true
        running: test.shell !== null && test.shell.visible && test.shell.uiModel.fileDialogSmokeCommand.length > 0
        onTriggered: {
            try {
                var command = test.shell.uiModel.fileDialogSmokeCommand;
                if (command.length === 0) return;
                if (command !== test.issued) test.issue(command);
                if (test.pending === "") return;
                test.waited += 1;
                if (test.waited > 250) throw new Error("dialog command timeout: " + test.issued);
                if (test.pending === "capture-wait" || test.pending === "capture-open-wait") return;
                if (test.pending === "closed-wait") {
                    if (!test.picker.dialogVisible) test.acknowledge();
                } else test.advance();
            } catch (error) {
                stop(); Julia.smoke_failed("FileDialog smoke: " + error.message);
            }
        }
    }
}
