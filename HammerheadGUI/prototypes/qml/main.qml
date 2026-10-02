import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Makie
import jlqml

ApplicationWindow {
    id: root
    visible: offscreenDisplay
    width: 1250; height: 820
    title: "Hammerhead: isolated Qt6 shell evaluation"
    color: "#f3f4f6"
    palette.window: "#f3f4f6"
    palette.windowText: "#17212b"
    palette.base: "white"
    palette.text: "#17212b"
    palette.button: "#e5e7eb"
    palette.buttonText: "#17212b"
    property bool separate: false
    property bool viewportOpen: true
    property bool drawing: false
    property int lifecycleStep: 0
    onClosing: Julia.shutdown_prototype()

    menuBar: MenuBar {
        Menu {
            title: "&View"
            Action { text: "Separate visualization window"; checkable: true
                checked: root.separate; onTriggered: root.separate = checked }
            Action { text: "Close / reopen visualization"; shortcut: "Ctrl+W"
                onTriggered: root.viewportOpen = !root.viewportOpen }
        }
    }
    Shortcut { sequence: "Right"; onActivated: Julia.navigate_frame(model.frame + 1) }
    Shortcut { sequence: "Left"; onActivated: Julia.navigate_frame(model.frame - 1) }
    Shortcut { sequence: "Escape"; onActivated: Julia.cancel_batch() }
    Shortcut { sequence: "Ctrl+Return"; onActivated: Julia.run_batch() }

    SplitView {
        id: shell
        anchors.fill: parent; orientation: Qt.Horizontal
        ScrollView {
            SplitView.preferredWidth: 350; SplitView.minimumWidth: 330
            padding: 16
            background: Rectangle { color: "white" }
            ColumnLayout {
                width: 304; spacing: 12
                Label { text: "Planar analysis settings"; font.pixelSize: 23 }
                Label { text: "Synthetic paired frames: 96 x 96 px\nRaw planar PIV first; no resume" }
                Label { text: "Window schedule" }
                TextField {
                    id: scheduleInput; text: "32"; Layout.fillWidth: true
                    Accessible.name: "PIV window schedule"
                    onEditingFinished: Julia.change_schedule(text)
                }
                Label {
                    text: model.scheduleError; color: "#b91c1c"; wrapMode: Text.Wrap
                    Layout.fillWidth: true; visible: text.length > 0
                }
                RowLayout {
                    Button { text: "Run demo"; enabled: !model.running; onClicked: Julia.run_batch() }
                    Button { text: "Cancel"; enabled: model.running; onClicked: Julia.cancel_batch() }
                }
                Label { text: model.status; wrapMode: Text.Wrap; Layout.fillWidth: true }
                Label { text: "Completed result file (lazy index)" }
                TextField {
                    id: pathInput; placeholderText: "C:/data/results.jld2"; Layout.fillWidth: true
                    Accessible.name: "Completed result file path"
                    onAccepted: Julia.open_results(text)
                }
                Button { text: "Open completed file"; onClicked: Julia.open_results(pathInput.text) }
                Label {
                    text: model.openError; color: "#b91c1c"; wrapMode: Text.Wrap
                    Layout.fillWidth: true; visible: text.length > 0
                }
                Label { text: "Frame " + model.frame + " / " + model.count }
                Slider {
                    Layout.fillWidth: true; from: 1; to: Math.max(1, model.count)
                    stepSize: 1; value: model.frame
                    Accessible.name: "Result frame"
                    onMoved: Julia.navigate_frame(Math.round(value))
                }
                CheckBox { text: "Draw exclusion mask"; checked: root.drawing
                    onToggled: { root.drawing = checked; Julia.set_mask_mode(checked) } }
                Button { text: "Close mask polygon"; onClicked: Julia.close_mask() }
                Label { text: model.selection; wrapMode: Text.Wrap; Layout.fillWidth: true }
                Label {
                    text: "Wheel: zoom; drag: pan\nArrows: frames; Esc: cancel\nCtrl+W: close / reopen viewport"
                    wrapMode: Text.Wrap; Layout.fillWidth: true
                }
                Label {
                    text: bridgeEnabled ? "QMLMakie OpenGL bridge" : "Static image fallback: refreshed after controller actions"
                    wrapMode: Text.Wrap; color: "#555"; Layout.fillWidth: true
                }
            }
        }
        Loader {
            id: integrated
            SplitView.fillWidth: true
            active: root.viewportOpen && !root.separate
            sourceComponent: viewportComponent
        }
    }

    Window {
        id: visualizationWindow
        visible: offscreenDisplay && root.separate && root.viewportOpen
        width: 900; height: 650; title: "Hammerhead visualization (same controllers)"
        onClosing: function(close) { root.viewportOpen = false; close.accepted = false }
        Loader { anchors.fill: parent; active: root.viewportOpen && root.separate
            sourceComponent: viewportComponent }
    }
    Component {
        id: viewportComponent
        Item {
            Component.onCompleted: Julia.viewport_created()
            Loader { anchors.fill: parent
                sourceComponent: bridgeEnabled ? nativeViewport : fallbackViewport }
        }
    }
    Component {
        id: nativeViewport
        MakieArea { scene: plot; focus: true }
    }
    Component {
        id: fallbackViewport
        Rectangle {
            clip: true; color: "#fafafa"
            Image {
                id: preview; x: (parent.width - width) / 2; y: (parent.height - height) / 2
                width: parent.width; height: parent.height
                source: model.preview; fillMode: Image.PreserveAspectFit; cache: false
                WheelHandler { target: preview; property: "scale" }
                DragHandler { target: preview }
                TapHandler {
                    onTapped: function(eventPoint) {
                        // Pick maps through the saved Makie axis rectangle, not figure margins.
                        Julia.pick_preview(eventPoint.position.x / preview.width,
                                           eventPoint.position.y / preview.height, root.drawing)
                    }
                }
            }
        }
    }
    // Deterministic lifecycle smoke, also exercises QML -> Julia form callbacks.
    Timer {
        interval: 120; running: smokeMode; repeat: true
        onTriggered: {
            root.lifecycleStep += 1
            if (root.lifecycleStep === 1) Julia.change_schedule("invalid")
            if (root.lifecycleStep === 2) Julia.change_schedule("32")
            if (root.lifecycleStep === 3) Julia.open_results("missing-results.jld2")
            if (root.lifecycleStep === 4) root.viewportOpen = false
            if (root.lifecycleStep === 5) root.viewportOpen = true
            if (root.lifecycleStep === 6) root.separate = true
            if (root.lifecycleStep === 7) root.viewportOpen = false
            if (root.lifecycleStep === 8) root.viewportOpen = true
            if (root.lifecycleStep === 9) root.separate = false
            if (root.lifecycleStep === 10) Julia.run_batch()
            if (root.lifecycleStep === 11) Julia.cancel_batch()
            if (root.lifecycleStep === 12) Julia.lifecycle_complete()
        }
    }
    Timer {
        interval: 3500; running: smokeMode; repeat: false
        onTriggered: shell.grabToImage(function(result) {
            Julia.record_capture(result.saveToFile(capturePath))
        })
    }
}
