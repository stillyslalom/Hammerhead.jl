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
    font.family: shellFont.status === FontLoader.Ready ? shellFont.name : ""
    palette.window: "#f3f4f6"
    palette.windowText: "#17212b"
    palette.base: "white"
    palette.text: "#17212b"
    palette.button: "#e5e7eb"
    palette.buttonText: "#17212b"
    property bool separate: false
    property bool viewportOpen: true
    property bool viewportLoaded: true
    property bool actualSeparate: false
    property bool transitionPending: false
    property bool transitionsReady: false
    property var uiModel: model
    property bool drawing: false
    property int lifecycleStep: 0
    onClosing: Julia.shutdown_prototype()
    function transitionViewport() {
        if (!root.transitionsReady || root.transitionPending) return;
        root.transitionPending = true;
        // Dispose Julia callbacks before destroying the QML item. Acknowledge
        // application release only; this makes no Qt GL-context cleanup claim.
        Julia.release_viewport(); root.viewportLoaded = false;
        Qt.callLater(function() {
            Julia.detach_viewport();
            root.actualSeparate = root.separate;
            if (root.viewportOpen) { Julia.prepare_viewport(); root.viewportLoaded = true; }
            root.transitionPending = false;
        });
    }
    onViewportOpenChanged: transitionViewport()
    onSeparateChanged: transitionViewport()
    Component.onCompleted: root.transitionsReady = true
    FontLoader {
        id: shellFont; source: fontSource
        onStatusChanged: {
            if (status === FontLoader.Ready) Julia.record_font_family(name);
            if (status === FontLoader.Error) Julia.smoke_failed("Qt FontLoader rejected the configured font file");
        }
    }

    menuBar: MenuBar {
        Menu {
            title: "&View"
            Action { text: "Separate visualization window"; checkable: true
                checked: root.separate; onTriggered: root.separate = checked }
            Action { text: "Close / reopen visualization"; shortcut: "Ctrl+W"
                onTriggered: root.viewportOpen = !root.viewportOpen }
        }
    }
    Shortcut { sequence: "Right"; onActivated: Julia.navigate_frame(root.uiModel.frame + 1) }
    Shortcut { sequence: "Left"; onActivated: Julia.navigate_frame(root.uiModel.frame - 1) }
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
                    text: root.uiModel.scheduleError; color: "#b91c1c"; wrapMode: Text.Wrap
                    Layout.fillWidth: true; visible: text.length > 0
                }
                RowLayout {
                    Button { text: "Run demo"; enabled: !root.uiModel.running; onClicked: Julia.run_batch() }
                    Button { text: "Cancel"; enabled: root.uiModel.running; onClicked: Julia.cancel_batch() }
                }
                Label { text: root.uiModel.status; wrapMode: Text.Wrap; Layout.fillWidth: true }
                Label { text: "Completed result file (lazy index)" }
                TextField {
                    id: pathInput; placeholderText: "C:/data/results.jld2"; Layout.fillWidth: true
                    Accessible.name: "Completed result file path"
                    onAccepted: Julia.open_results(text)
                }
                Button { text: "Open completed file"; onClicked: Julia.open_results(pathInput.text) }
                Label {
                    text: root.uiModel.openError; color: "#b91c1c"; wrapMode: Text.Wrap
                    Layout.fillWidth: true; visible: text.length > 0
                }
                Label { text: "Frame " + root.uiModel.frame + " / " + root.uiModel.count }
                Slider {
                    Layout.fillWidth: true; from: 1; to: Math.max(1, root.uiModel.count)
                    stepSize: 1; value: root.uiModel.frame
                    Accessible.name: "Result frame"
                    onMoved: Julia.navigate_frame(Math.round(value))
                }
                CheckBox { text: "Draw exclusion mask"; checked: root.drawing
                    onToggled: { root.drawing = checked; Julia.set_mask_mode(checked) } }
                Button { text: "Close mask polygon"; onClicked: Julia.close_mask() }
                Label { text: root.uiModel.selection; wrapMode: Text.Wrap; Layout.fillWidth: true }
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
            active: root.viewportLoaded && !root.actualSeparate
            sourceComponent: viewportComponent
        }
    }

    Window {
        id: visualizationWindow
        visible: offscreenDisplay && root.actualSeparate && root.viewportLoaded
        width: 900; height: 650; title: "Hammerhead visualization (same controllers)"
        onClosing: function(close) { root.viewportOpen = false; close.accepted = false }
        Loader { anchors.fill: parent; active: root.viewportLoaded && root.actualSeparate
            sourceComponent: viewportComponent }
    }
    Component {
        id: viewportComponent
        Item {
            // One stable figure value per QML item; no shared scene is reused
            // across native screens and no context-property history accumulates.
            property var ownedPlot: bridgeEnabled ? Julia.viewport_scene() : null
            Component.onCompleted: Julia.viewport_created()
            Loader { anchors.fill: parent
                sourceComponent: bridgeEnabled ? nativeViewport : fallbackViewport
                property var currentPlot: parent.ownedPlot }
        }
    }
    Component {
        id: nativeViewport
        MakieArea { anchors.fill: parent; scene: parent.currentPlot; focus: true }
    }
    Component {
        id: fallbackViewport
        Rectangle {
            clip: true; color: "#fafafa"
            Image {
                id: preview; x: (parent.width - width) / 2; y: (parent.height - height) / 2
                width: parent.width; height: parent.height
                source: root.uiModel.preview; fillMode: Image.PreserveAspectFit; cache: false
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
            if (root.transitionPending) return;
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
        interval: 350; running: smokeMode; repeat: true
        onTriggered: {
            if (root.lifecycleStep < 12 || root.transitionPending || root.uiModel.running || shellFont.status !== FontLoader.Ready) return;
            stop();
            shell.grabToImage(function(result) {
                Julia.record_capture(result.saveToFile(capturePath))
            });
        }
    }
    Timer { interval: 120000; running: smokeMode; repeat: false
        onTriggered: Julia.smoke_failed("software lifecycle smoke timed out awaiting transition/completion") }
}
