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
    property int acknowledgedTransition: 0
    property var uiModel: model
    property bool drawing: false
    property int lifecycleStep: 0
    onClosing: Julia.shutdown_prototype()
    function transitionViewport() {
        if (!root.transitionsReady || root.transitionPending) return;
        root.transitionPending = true;
        if (!bridgeEnabled) {
            root.viewportLoaded = false;
            Julia.queue_transition(root.viewportOpen,root.separate);
            return;
        }
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
    Timer { interval: 40; running: !bridgeEnabled; repeat: true
        onTriggered: {
            if (!root.transitionPending || root.uiModel.transitionAck === root.acknowledgedTransition) return;
            root.acknowledgedTransition = root.uiModel.transitionAck;
            root.actualSeparate = root.uiModel.transitionSeparate;
            root.viewportLoaded = root.uiModel.transitionOpen;
            root.transitionPending = false;
        }
    }
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
    Shortcut { sequence: "Escape"; onActivated: analysisLane.currentIndex === 1 ? Julia.cancel_saved() : Julia.cancel_batch() }
    Shortcut { sequence: "Ctrl+Return"; onActivated: analysisLane.currentIndex === 1 ? Julia.replay_saved(outputInput.text, historyInput.text, allowOverride.checked) : Julia.run_batch() }

    SplitView {
        id: shell
        anchors.fill: parent; orientation: Qt.Horizontal
        ScrollView {
            SplitView.preferredWidth: 350; SplitView.minimumWidth: 330
            padding: 16
            background: Rectangle { color: "white" }
            ColumnLayout {
                width: 304; spacing: 12
                ComboBox { id: analysisLane; model: ["Synthetic demo", "Saved experiment"]
                    currentIndex: experimentSmoke ? 1 : 0; Layout.fillWidth: true }
                ColumnLayout {
                visible: analysisLane.currentIndex === 0; Layout.fillWidth: true
                Label { text: "Planar demo settings"; font.pixelSize: 23 }
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
                }
                ColumnLayout {
                    visible: analysisLane.currentIndex === 1; Layout.fillWidth: true; spacing: 6
                    Label { text: "Saved planar experiment"; font.pixelSize: 22 }
                    TextField { id: experimentInput; Layout.fillWidth: true
                        text: root.uiModel.fixtureRecord; placeholderText: "Saved experiment .jld2"
                        Accessible.name: "Saved experiment path" }
                    Button { text: "Open saved experiment"; enabled: !root.uiModel.experimentRunning
                        onClicked: Julia.open_saved(experimentInput.text) }
                    TextField { id: outputInput; Layout.fillWidth: true; text: root.uiModel.experimentOutput
                        placeholderText: "Result output .jld2"; Accessible.name: "Replay result output" }
                    TextField { id: historyInput; Layout.fillWidth: true; text: root.uiModel.experimentHistory
                        placeholderText: "Run record .jld2 (optional)"; Accessible.name: "Replay run record" }
                    CheckBox { id: allowOverride; text: "Allow environment change (recorded)"
                        checked: root.uiModel.experimentAllow; enabled: !root.uiModel.experimentRunning }
                    RowLayout {
                        Button { text: "Replay saved recipe"; enabled: !root.uiModel.experimentRunning
                            onClicked: Julia.replay_saved(outputInput.text, historyInput.text, allowOverride.checked) }
                        Button { text: "Cancel"; enabled: root.uiModel.experimentRunning; onClicked: Julia.cancel_saved() }
                    }
                    Label { text: root.uiModel.experimentWritten; font.bold: true }
                    Label { text: root.uiModel.experimentStatus; wrapMode: Text.WrapAnywhere; Layout.fillWidth: true }
                    Label { text: root.uiModel.experimentError; color: "#b91c1c"; wrapMode: Text.WrapAnywhere
                        Layout.fillWidth: true; visible: text.length > 0 }
                    Button { text: "Inspect completed run"; enabled: !root.uiModel.experimentRunning
                        onClicked: Julia.inspect_saved() }
                    ComboBox { model: ["Complete recipe", "Run history"]; Layout.fillWidth: true
                        onActivated: Julia.saved_section(currentIndex === 1) }
                    RowLayout {
                        Button { text: "Previous"; onClicked: Julia.saved_page(-1) }
                        Label { text: root.uiModel.experimentPages }
                        Button { text: "Next"; onClicked: Julia.saved_page(1) }
                    }
                    ScrollView { Layout.fillWidth: true; Layout.preferredHeight: 140
                        TextArea { text: root.uiModel.experimentText; readOnly: true; selectByMouse: true
                            wrapMode: TextEdit.WrapAnywhere; font.pixelSize: 12 } }
                }
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
                CheckBox { text: "Draw demo exclusion mask"; checked: root.drawing
                    enabled: root.uiModel.demoDisplayed
                    onToggled: { root.drawing = checked; Julia.set_mask_mode(checked) } }
                Button { text: "Close demo mask polygon"; enabled: root.uiModel.demoDisplayed; onClicked: Julia.close_mask() }
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
            id: viewportItem
            Rectangle { anchors.fill: parent; color: "white" }
            // One stable figure value per QML item; no shared scene is reused
            // across native screens and no context-property history accumulates.
            property var ownedPlot: bridgeEnabled ? Julia.viewport_scene() : null
            Component.onCompleted: Julia.viewport_created()
            ColumnLayout { anchors.fill: parent
            Label { text: root.uiModel.activeIdentity + "\n" + root.uiModel.displayedIdentity
                wrapMode: Text.WrapAnywhere; Layout.fillWidth: true; font.pixelSize: 12; padding: 8
                color: "#17212b"; background: Rectangle { color: "#f8fafc" } }
            Loader { Layout.fillWidth: true; Layout.fillHeight: true
                sourceComponent: bridgeEnabled ? nativeViewport : fallbackViewport
                property var currentPlot: viewportItem.ownedPlot }
            }
        }
    }
    Component {
        id: nativeViewport
        MakieArea { anchors.fill: parent; scene: parent.currentPlot; focus: true; visible: root.uiModel.renderAvailable }
    }
    Component {
        id: fallbackViewport
        Rectangle {
            clip: true; color: "#fafafa"
            Image {
                id: preview; x: (parent.width - width) / 2; y: (parent.height - height) / 2
                width: parent.width; height: parent.height
                source: root.uiModel.preview; fillMode: Image.PreserveAspectFit; cache: false
                visible: root.uiModel.renderAvailable
                WheelHandler { target: preview; property: "scale" }
                DragHandler { target: preview }
                TapHandler {
                    onTapped: function(eventPoint) {
                        // Pick maps through the saved Makie axis rectangle, not figure margins.
                        Julia.pick_preview((eventPoint.position.x - (preview.width - preview.paintedWidth) / 2) / preview.paintedWidth,
                                           (eventPoint.position.y - (preview.height - preview.paintedHeight) / 2) / preview.paintedHeight, root.drawing)
                    }
                }
            }
        }
    }
    // Deterministic lifecycle smoke, also exercises QML -> Julia form callbacks.
    Timer {
        interval: 120; running: smokeMode && !experimentSmoke; repeat: true
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
    Timer { interval: 120; running: smokeMode && experimentSmoke; repeat: true
        onTriggered: {
            if (root.transitionPending) return;
            var step = Julia.experiment_smoke_step();
            if (step === 1) root.viewportOpen = false;
            if (step === 2) root.viewportOpen = true;
            if (step === 3) root.separate = true;
            if (step === 4) root.separate = false;
            if (step === 5) root.viewportOpen = false;
            if (step === 6) root.viewportOpen = true;
            if (step === 7) { root.lifecycleStep = 12; Julia.lifecycle_complete(); stop(); }
        }
    }
    Timer {
        interval: 350; running: smokeMode; repeat: true
        onTriggered: {
            if (root.lifecycleStep < 12 || root.transitionPending || root.uiModel.running || root.uiModel.experimentRunning || shellFont.status !== FontLoader.Ready) return;
            stop();
            shell.grabToImage(function(result) {
                Julia.record_capture(result.saveToFile(capturePath))
            });
        }
    }
    Timer { interval: 120000; running: smokeMode; repeat: false
        onTriggered: Julia.smoke_failed("software lifecycle smoke timed out awaiting transition/completion") }
}
