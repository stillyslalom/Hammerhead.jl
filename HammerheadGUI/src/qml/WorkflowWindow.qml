// Window chrome shared by the workflow windows: header toolbar, step rail,
// the step pages (children of a window go into the page stack, in the order
// of `stepKeys`), the persistent viewer canvas and its pop-out window, the
// tick timer, and the settings dialogs. All state lives in Julia: `app`
// (property map), `stepModel` (item model); actions call Julia.hh_* functions.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Dialogs
import Makie
import jlqml

ApplicationWindow {
    id: win
    title: app.title
    width: 1480
    height: 920
    minimumWidth: 1100
    minimumHeight: 680
    visible: true

    property bool quitting: false
    property var stepKeys: []
    // stereo windows: the pair bar gains the camera switch
    property bool stereo: false
    default property alias pages: stack.data

    // Give a canvas keyboard focus when the pointer enters it, unless a text
    // field is being edited (its typing would move to the canvas).
    function focusCanvas(canvas, hovered) {
        if (!hovered || canvas.activeFocus) return
        const item = canvas.Window.activeFocusItem
        if (item && item.hasOwnProperty("cursorPosition")) return
        canvas.forceActiveFocus(Qt.MouseFocusReason)
    }

    // Closing the main window ends the session (the pop-out window included).
    onClosing: { quitting = true; Qt.quit() }

    // Pair and frame navigation. Text fields keep the keys they use (Qt
    // offers them the key first), so typing is unaffected.
    Shortcut { sequences: ["Left"]; onActivated: Julia.hh_step_pair(-1) }
    Shortcut { sequences: ["Right"]; onActivated: Julia.hh_step_pair(1) }
    Shortcut { sequences: ["A", "Shift+Left"]; onActivated: Julia.hh_show_frame("a") }
    Shortcut { sequences: ["B", "Shift+Right"]; onActivated: Julia.hh_show_frame("b") }

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 8
            anchors.rightMargin: 8
            spacing: 4
            ToolButton { text: "Open settings…"; onClicked: openSettingsDialog.open() }
            ToolButton { text: "Save settings…"; onClicked: saveSettingsDialog.open() }
            Item { Layout.fillWidth: true }
            Label {
                text: app.status
                elide: Text.ElideRight
                Layout.maximumWidth: 560
                opacity: 0.75
            }
            ToolSeparator {}
            ToolButton {
                text: app.popped ? "Dock viewer" : "Pop out viewer"
                onClicked: Julia.hh_toggle_popout()
            }
        }
    }

    footer: PairBar { stereo: win.stereo }

    SplitView {
        id: body
        anchors.fill: parent
        orientation: Qt.Horizontal

        Pane {
            SplitView.preferredWidth: 760
            SplitView.minimumWidth: 600
            padding: 0
            clip: true

            RowLayout {
                anchors.fill: parent
                spacing: 0

                // the steps, with the app's mark at the foot of the rail (the
                // title bar already shows the icon at the top left)
                ColumnLayout {
                    // a nested layout fills the row by default; the rail must not
                    Layout.fillWidth: false
                    Layout.preferredWidth: 200
                    Layout.maximumWidth: 200
                    Layout.fillHeight: true
                    spacing: 0
                    ListView {
                        id: rail
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.topMargin: 8
                        model: stepModel
                        interactive: false
                        spacing: 2
                        delegate: StepDelegate {
                            width: rail.width
                            stepNumber: index + 1
                            current: model.key === app.step
                        }
                    }
                    RowLayout {
                        Layout.leftMargin: 14
                        Layout.bottomMargin: 12
                        spacing: 8
                        opacity: 0.8
                        Image {
                            source: "icons/hammerhead.svg"
                            sourceSize.width: 28
                            sourceSize.height: 28
                        }
                        Label { text: "Hammerhead"; font.weight: Font.DemiBold }
                    }
                }

                ToolSeparator { orientation: Qt.Vertical; Layout.fillHeight: true }

                StackLayout {
                    id: stack
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    // pages scroll; their content must not widen the pane
                    Layout.preferredWidth: 0
                    Layout.minimumWidth: 0
                    currentIndex: Math.max(0, win.stepKeys.indexOf(app.step))
                }
            }
        }

        // The canvas items live as long as their windows: QMLMakie canvases
        // must not be destroyed while the application runs.
        Item {
            SplitView.fillWidth: true
            SplitView.minimumWidth: 400
            MakieArea {
                id: mainCanvas
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                anchors.right: mainTools.left
                scene: app.main
                visible: !app.popped
                // QMLMakie passes clicks on only while the canvas has focus:
                // take it when the pointer arrives, so the first click counts
                HoverHandler { onHoveredChanged: win.focusCanvas(mainCanvas, hovered) }
            }
            ViewerToolbar {
                id: mainTools
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                visible: !app.popped
                onSaveView: saveViewDialog.open()
            }
            Label {
                anchors.centerIn: parent
                visible: app.popped
                text: "The viewer is in its own window"
                opacity: 0.7
            }
        }
    }

    Window {
        id: popWindow
        title: "Hammerhead viewer"
        width: 1000
        height: 900
        visible: app.popped
        // Closing the pop-out docks the viewer, except while the app quits
        // (a rejected close would cancel Qt's quit).
        onClosing: (close) => {
            if (!win.quitting) {
                close.accepted = false
                Julia.hh_toggle_popout()
            }
        }
        MakieArea {
            id: popCanvas
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.right: popTools.left
            scene: app.pop
            HoverHandler { onHoveredChanged: win.focusCanvas(popCanvas, hovered) }
        }
        ViewerToolbar {
            id: popTools
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            onSaveView: saveViewDialog.open()
        }
    }

    FileDialog {
        id: saveViewDialog
        title: "Save the viewer as an image"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "png"
        nameFilters: ["PNG image (*.png)"]
        onAccepted: {
            const target = selectedFile
            const canvas = app.popped ? popCanvas : mainCanvas
            canvas.grabToImage(function (img) { img.saveToFile(target) })
        }
    }

    Timer {
        interval: 16
        running: true
        repeat: true
        onTriggered: {
            const r = Julia.hh_tick()
            if (app.switchQuestion !== "" && !switchDialog.visible) switchDialog.open()
            if (app.grabPath !== "") {
                const path = app.grabPath
                Julia.hh_grab_started()
                body.grabToImage(function (img) { img.saveToFile(path) })
            }
            if (r === 2) {
                win.close()
            } else if (r === 1) {
                mainCanvas.update()
                popCanvas.update()
            }
        }
    }

    // Asked before a change of recording type discards the session's work.
    Dialog {
        id: switchDialog
        title: "Change the recording type"
        modal: true
        anchors.centerIn: Overlay.overlay
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 480
        contentItem: Label {
            text: app.switchQuestion
            wrapMode: Text.WordWrap
        }
        Component.onCompleted: standardButton(Dialog.Ok).text = "Discard and switch"
        onAccepted: Julia.hh_confirm_switch()
        // Cancel, Escape, or a click outside
        onClosed: if (app.switchQuestion !== "") Julia.hh_cancel_switch()
    }

    FileDialog {
        id: openSettingsDialog
        title: "Open settings"
        nameFilters: ["Settings or results (*.toml *.jld2)", "All files (*)"]
        onAccepted: Julia.hh_open_settings(selectedFile.toString())
    }
    FileDialog {
        id: saveSettingsDialog
        title: "Save settings"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "toml"
        nameFilters: ["Settings (*.toml)", "Settings in a JLD2 file (*.jld2)"]
        onAccepted: Julia.hh_save_settings(selectedFile.toString())
    }
}
