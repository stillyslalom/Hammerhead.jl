import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Makie
import jlqml

ApplicationWindow {
    id: root
    objectName: "prototypeShell"
    visible: applicationMode ? applicationVisible : offscreenDisplay
    width: glfwPlotMode ? 650 : 1250; height: glfwPlotMode ? 900 : 820
    title: applicationMode ? "Hammerhead — Experimental Qt GUI" : "Hammerhead: isolated Qt6 shell evaluation"
    color: "#f3f4f6"
    font.family: shellFont.status === FontLoader.Ready ? shellFont.name : ""
    font.pixelSize: 14
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
    property bool sidebarSmallCaptured: false
    property string workerIssuedCommand: ""
    property int workerCommandStep: 0
    property alias nativeResultPicker: pathInput
    property alias savedRecordPicker: experimentInput
    property alias replayOutputPicker: outputInput
    property alias runHistoryPicker: historyInput
    property alias analysisLaneControl: analysisLane
    property alias savedSectionControl: savedControlsSection
    property alias captureSurface: shell
    onClosing: {
        pathInput.dismiss(); experimentInput.dismiss(); outputInput.dismiss(); historyInput.dismiss();
        Julia.shutdown_prototype();
    }
    Loader {
        active: fileDialogSmokeMode
        source: active ? "file_dialog_shell_smoke.qml" : ""
        onLoaded: item.shell = root
    }
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
            if (root.uiModel.transitionAck === root.acknowledgedTransition) return;
            if (!root.transitionPending && !glfwPlotMode) return;
            // A GLFW close is unsolicited. Update controls under the same guard
            // as an acknowledged request, so one Ctrl+W reopens the window.
            root.transitionPending = true;
            root.acknowledgedTransition = root.uiModel.transitionAck;
            if (glfwPlotMode) {
                root.viewportOpen = root.uiModel.transitionOpen;
                root.separate = root.uiModel.transitionSeparate;
            }
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
            Action { text: glfwPlotMode ? "Separate interactive scientific plot" : "Separate visualization window"; checkable: true; enabled: !glfwPlotMode && !root.uiModel.fileDialogOpen
                checked: glfwPlotMode || root.separate; onTriggered: { if (!glfwPlotMode) root.separate = checked; } }
            Action { text: "Close / reopen visualization"; shortcut: "Ctrl+W"
                enabled: !root.uiModel.fileDialogOpen
                onTriggered: root.viewportOpen = !root.viewportOpen }
        }
    }
    Shortcut { sequence: "Right"; enabled: !root.uiModel.fileDialogOpen; onActivated: Julia.navigate_frame(root.uiModel.frame + 1) }
    Shortcut { sequence: "Left"; enabled: !root.uiModel.fileDialogOpen; onActivated: Julia.navigate_frame(root.uiModel.frame - 1) }
    Shortcut { sequence: "Escape"; enabled: !root.uiModel.fileDialogOpen; onActivated: analysisLane.currentIndex === 1 ? Julia.cancel_saved() : Julia.cancel_batch() }
    Shortcut { sequence: "Ctrl+Return"; enabled: !root.uiModel.fileDialogOpen; onActivated: analysisLane.currentIndex === 1 ? Julia.replay_saved(outputInput.text, historyInput.text, allowOverride.checked) : Julia.run_batch() }

    SplitView {
        id: shell
        anchors.fill: parent; orientation: Qt.Horizontal
        Item {
            id: sidebarPanel
            objectName: "sidebarPanel"
            SplitView.preferredWidth: 350; SplitView.minimumWidth: 300
            Rectangle { anchors.fill: parent; color: "white" }
            ColumnLayout {
            anchors.fill: parent; anchors.margins: 16; spacing: 8
            ComboBox { id: analysisLane; model: ["Synthetic demo", "Saved experiment"]
                currentIndex: initialSavedLane || experimentSmoke ? 1 : 0; Layout.fillWidth: true }
            ComboBox { id: savedControlsSection; objectName: "savedControlsSection"
                model: ["Files", "Replay", "Inspection"]; Layout.fillWidth: true
                visible: analysisLane.currentIndex === 1 }
        ScrollView {
            id: sidebarScroll
            objectName: "sidebarScroll"
            Layout.fillWidth: true; Layout.fillHeight: true
            contentWidth: availableWidth
            padding: 0
            background: Rectangle { color: "white" }
            ColumnLayout {
                id: sidebarColumn
                objectName: "sidebarColumn"
                width: sidebarScroll.availableWidth; spacing: 8
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
                    Button { text: "Run demo"; enabled: !root.uiModel.running && !root.uiModel.fileDialogOpen; onClicked: Julia.run_batch() }
                    Button { text: "Cancel"; enabled: root.uiModel.running; onClicked: Julia.cancel_batch() }
                }
                Label { text: root.uiModel.status; wrapMode: Text.Wrap; Layout.fillWidth: true }
                Label { text: "Completed result file (lazy index)" }
                FilePathPicker {
                    id: pathInput; placeholderText: "C:/data/results.jld2"; Layout.fillWidth: true
                    text: root.uiModel.initialResult
                    purpose: "result"; fieldName: "nativeResultPathField"; dialogTitle: "Completed result file"
                    forceNonNative: forceNonNativeDialogs
                    dialogsEnabled: !root.uiModel.running && !root.uiModel.experimentRunning && !root.uiModel.fileDialogOpen
                    onSubmitted: Julia.open_results(path)
                }
                Button { text: "Open completed file"; enabled: !root.uiModel.fileDialogOpen; onClicked: Julia.open_results(pathInput.text) }
                }
                ColumnLayout {
                    visible: analysisLane.currentIndex === 1; Layout.fillWidth: true; spacing: 6
                    Label { text: "Saved planar experiment"; font.pixelSize: 20; Layout.fillWidth: true; wrapMode: Text.Wrap }
                    ColumnLayout {
                    visible: savedControlsSection.currentIndex === 0; Layout.fillWidth: true; spacing: 6
                    Label { text: "Saved recipe"; font.pixelSize: 12; color: "#555"; Layout.fillWidth: true }
                    FilePathPicker { id: experimentInput; Layout.fillWidth: true
                        text: root.uiModel.fixtureRecord; placeholderText: "Saved experiment .jld2"
                        purpose: "record"; fieldName: "savedRecordPathField"; dialogTitle: "Saved experiment"
                        forceNonNative: forceNonNativeDialogs
                        dialogsEnabled: !root.uiModel.running && !root.uiModel.experimentRunning && !root.uiModel.fileDialogOpen }
                    Button { text: "Open saved experiment"; enabled: !root.uiModel.experimentRunning && !root.uiModel.fileDialogOpen
                        onClicked: Julia.open_saved(experimentInput.text) }
                    Label { text: "Result output"; font.pixelSize: 12; color: "#555"; Layout.fillWidth: true }
                    FilePathPicker { id: outputInput; Layout.fillWidth: true; text: root.uiModel.experimentOutput
                        placeholderText: "Result output .jld2"; purpose: "output"; saveMode: true
                        fieldName: "replayOutputPathField"; dialogTitle: "Replay result output"
                        forceNonNative: forceNonNativeDialogs
                        dialogsEnabled: !root.uiModel.running && !root.uiModel.experimentRunning && !root.uiModel.fileDialogOpen }
                    Label { text: "Run history (optional)"; font.pixelSize: 12; color: "#555"; Layout.fillWidth: true }
                    FilePathPicker { id: historyInput; Layout.fillWidth: true; text: root.uiModel.experimentHistory
                        placeholderText: "Run record .jld2 (optional)"; purpose: "history"; saveMode: true
                        fieldName: "runHistoryPathField"; dialogTitle: "Replay run record"
                        forceNonNative: forceNonNativeDialogs
                        dialogsEnabled: !root.uiModel.running && !root.uiModel.experimentRunning && !root.uiModel.fileDialogOpen }
                    Label { text: "Complete saved settings stay read-only. Paths and options describe the next replay."; Layout.fillWidth: true; wrapMode: Text.Wrap }
                    }
                    ColumnLayout {
                    visible: savedControlsSection.currentIndex === 1; Layout.fillWidth: true; spacing: 6
                    CheckBox { id: allowOverride; text: "Allow environment change (recorded)"
                        checked: root.uiModel.experimentAllow; enabled: !root.uiModel.experimentRunning }
                        Button { id: replayButton; objectName: "replaySavedButton"; text: "Replay saved recipe"; enabled: !root.uiModel.experimentRunning && !root.uiModel.fileDialogOpen
                            onClicked: Julia.replay_saved(outputInput.text, historyInput.text, allowOverride.checked) }
                    Label { text: "Replay uses the saved settings. Cancel takes effect after the current pair finishes and cleanup completes; no resume."; Layout.fillWidth: true; wrapMode: Text.Wrap }
                    }
                    ColumnLayout {
                    visible: savedControlsSection.currentIndex === 2; Layout.fillWidth: true; spacing: 6
                    Button { text: "Inspect completed run"; enabled: !root.uiModel.experimentRunning && !root.uiModel.fileDialogOpen
                        onClicked: Julia.inspect_saved() }
                    ComboBox { model: ["Complete recipe", "Run history"]; Layout.fillWidth: true
                        onActivated: Julia.saved_section(currentIndex === 1) }
                    RowLayout {
                        Button { text: "Previous"; onClicked: Julia.saved_page(-1) }
                        Label { text: root.uiModel.experimentPages }
                        Button { text: "Next"; onClicked: Julia.saved_page(1) }
                    }
                    ScrollView { id: recipeScroll; objectName: "recipeScroll"; Layout.fillWidth: true; Layout.preferredHeight: 140
                        TextArea { id: recipeText; objectName: "recipeText"; text: root.uiModel.experimentText; readOnly: true; selectByMouse: true
                            wrapMode: TextEdit.WrapAnywhere; font.pixelSize: 12 } }
                    }
                }
                Label {
                    text: root.uiModel.openError; color: "#b91c1c"; wrapMode: Text.Wrap
                    Layout.fillWidth: true; visible: text.length > 0
                }
                Label { text: root.uiModel.fileDialogError; visible: text.length > 0
                    color: "#b91c1c"; wrapMode: Text.WrapAnywhere; Layout.fillWidth: true }
                ColumnLayout {
                visible: analysisLane.currentIndex === 0 || savedControlsSection.currentIndex === 2
                Layout.fillWidth: true; spacing: 6
                Label { text: "Frame " + root.uiModel.frame + " / " + root.uiModel.count }
                Slider {
                    Layout.fillWidth: true; from: 1; to: Math.max(1, root.uiModel.count)
                    stepSize: 1; value: root.uiModel.frame
                    Accessible.name: "Result frame"
                    onMoved: Julia.navigate_frame(Math.round(value))
                }
                CheckBox { text: "Draw demo exclusion mask"; checked: root.drawing
                    visible: analysisLane.currentIndex === 0
                    enabled: root.uiModel.demoDisplayed
                    onToggled: { root.drawing = checked; Julia.set_mask_mode(checked) } }
                Button { text: "Close demo mask polygon"; visible: analysisLane.currentIndex === 0; enabled: root.uiModel.demoDisplayed; onClicked: Julia.close_mask() }
                Label { text: root.uiModel.selection; wrapMode: Text.Wrap; Layout.fillWidth: true }
                Label {
                    text: glfwPlotMode ? "Scientific window: wheel zoom, right-drag pan\nControls: arrows change frames; Esc cancels\nCtrl+W closes / reopens visualization" : "Wheel: zoom; drag: pan\nArrows: frames; Esc: cancel\nCtrl+W: close / reopen viewport"
                    wrapMode: Text.Wrap; Layout.fillWidth: true
                }
                }
                Label {
                    id: sidebarBottom
                    objectName: "sidebarBottom"
                    text: glfwPlotMode ? "Separate interactive scientific plot" : bridgeEnabled ? "QMLMakie OpenGL bridge" : "Static image fallback: refreshed after controller actions"
                    wrapMode: Text.Wrap; color: "#555"; Layout.fillWidth: true
                }
            }
        }
        ColumnLayout {
            id: savedPersistent; objectName: "savedPersistent"
            visible: analysisLane.currentIndex === 1; Layout.fillWidth: true; spacing: 4
            Button { id: savedCancelButton; objectName: "savedCancelButton"; text: "Cancel replay"
                enabled: root.uiModel.experimentRunning; onClicked: Julia.cancel_saved() }
            Label { objectName: "savedWritten"; text: root.uiModel.experimentWritten; font.bold: true; Layout.fillWidth: true }
            Label { objectName: "savedStatus"; text: root.uiModel.experimentStatusPreview; Layout.fillWidth: true
                wrapMode: Text.WrapAnywhere; maximumLineCount: 2; elide: Text.ElideRight }
            Label { text: root.uiModel.experimentError.length > 0 ? "Error: full detail in inspection pages" : "Full status and paths in inspection pages"
                color: root.uiModel.experimentError.length > 0 ? "#b91c1c" : "#555"; font.pixelSize: 12; Layout.fillWidth: true; wrapMode: Text.Wrap }
        }
        }
        }
        Loader {
            id: integrated
            SplitView.fillWidth: true
            active: root.viewportLoaded && (!root.actualSeparate || glfwPlotMode)
            sourceComponent: viewportComponent
        }
    }
    // Opt-in baseline: measure scrolling before replacing tall controls. Direct
    // contentY positioning proves geometric reachability, not desktop wheel input.
    function sidebarProbe(stage) {
        var bottom = sidebarBottom.mapToItem(sidebarScroll, 0, 0);
        var persistent = savedPersistent.mapToItem(sidebarPanel, 0, 0);
        var innerMax = Math.max(0, recipeScroll.contentHeight - recipeScroll.availableHeight);
        Julia.record_sidebar_probe(stage, root.width, root.height,
            sidebarScroll.availableWidth, sidebarColumn.width,
            sidebarScroll.contentHeight, sidebarScroll.availableHeight,
            sidebarScroll.contentItem.contentY,
            Math.max(0, sidebarScroll.contentHeight - sidebarScroll.availableHeight),
            bottom.y, sidebarBottom.height, recipeScroll.contentHeight,
            recipeScroll.availableHeight, recipeScroll.contentItem.contentY, innerMax,
            bottom.y >= -1 && bottom.y + sidebarBottom.height <= sidebarScroll.height + 1,
            savedPersistent.visible && persistent.y >= -1 && persistent.y + savedPersistent.height <= sidebarPanel.height + 1,
            Math.abs(recipeScroll.contentItem.contentY-innerMax) <= 2);
    }
    Timer {
        property int probeStep: 0
        interval: 120; running: sidebarProbeMode; repeat: true
        onTriggered: {
            if (probeStep === 3 && !root.sidebarSmallCaptured) return;
            probeStep += 1;
            if (probeStep === 1) { root.width = 900; root.height = 600; savedControlsSection.currentIndex = 2; }
            if (probeStep === 2) {
                root.sidebarProbe("small_top");
                sidebarScroll.contentItem.contentY = Math.max(0, sidebarScroll.contentHeight-sidebarScroll.availableHeight);
                recipeScroll.contentItem.contentY = Math.max(0, recipeScroll.contentHeight-recipeScroll.availableHeight);
            }
            if (probeStep === 3) {
                root.sidebarProbe("small_bottom");
                shell.grabToImage(function(result) {
                    Julia.record_sidebar_capture("small", result.saveToFile(sidebarCapturePrefix + "-small.png"));
                    root.width = 1100; root.height = 800;
                    root.sidebarSmallCaptured = true;
                });
            }
            if (probeStep === 4) {
                sidebarScroll.contentItem.contentY = Math.max(0, sidebarScroll.contentHeight-sidebarScroll.availableHeight);
                recipeScroll.contentItem.contentY = Math.max(0, recipeScroll.contentHeight-recipeScroll.availableHeight);
            }
            if (probeStep === 5) {
                root.sidebarProbe("large_bottom");
                shell.grabToImage(function(result) { Julia.record_sidebar_capture("large", result.saveToFile(sidebarCapturePrefix + "-large.png")); });
                stop();
            }
        }
    }

    Window {
        id: visualizationWindow
        visible: offscreenDisplay && root.actualSeparate && root.viewportLoaded && !glfwPlotMode
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
                sourceComponent: glfwPlotMode ? separateGLFWStatus : bridgeEnabled ? nativeViewport : fallbackViewport
                property var currentPlot: viewportItem.ownedPlot }
            }
        }
    }
    Component {
        id: separateGLFWStatus
        Rectangle {
            color: "#f8fafc"
            ColumnLayout { anchors.fill: parent; anchors.margins: 12
                Label { Layout.fillWidth: true; wrapMode: Text.Wrap
                    text: "The interactive scientific plot is in a separate window. Closing it leaves processing running; Ctrl+W reopens it."
                }
                Label { Layout.fillWidth: true; wrapMode: Text.Wrap
                    text: "Frame " + root.uiModel.frame + " / " + root.uiModel.count + "\n\n" + root.uiModel.selection
                }
                Item { Layout.fillHeight: true }
                Label { Layout.fillWidth: true; wrapMode: Text.Wrap; color: "#555"
                    text: applicationMode ? "Use View to close or reopen the scientific plot." : "Hidden prototype capture; desktop input remains unverified."
                }
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
        interval: 120; running: smokeMode && !experimentSmoke && !workerSmoke; repeat: true
        onTriggered: {
            if (root.transitionPending) return;
            if (glfwPlotMode && root.uiModel.transitionAck !== root.acknowledgedTransition) return;
            root.lifecycleStep += 1
            if (root.lifecycleStep === 1) Julia.change_schedule("invalid")
            if (root.lifecycleStep === 2) Julia.change_schedule("32")
            if (root.lifecycleStep === 3) {
                Julia.open_results("missing-results.jld2");
                if (glfwPlotMode) {
                    // Wait for the owner before advancing another timer step
                    // within this same Qt event pass.
                    root.transitionPending = true;
                    Julia.simulate_glfw_close();
                }
            }
            if (root.lifecycleStep === 4) {
                if (glfwPlotMode) {
                    if (root.viewportOpen) Julia.smoke_failed("GLFW close was not acknowledged by controls");
                } else root.viewportOpen = false;
            }
            if (root.lifecycleStep === 5) {
                if (glfwPlotMode) root.viewportOpen = !root.viewportOpen;
                else root.viewportOpen = true;
            }
            if (root.lifecycleStep === 6) root.separate = true
            if (root.lifecycleStep === 7) root.viewportOpen = false
            if (root.lifecycleStep === 8) root.viewportOpen = true
            if (root.lifecycleStep === 9) root.separate = false
            if (root.lifecycleStep === 10) Julia.run_batch()
            if (root.lifecycleStep === 11) Julia.cancel_batch()
            if (root.lifecycleStep === 12) Julia.lifecycle_complete()
        }
    }
    Timer { interval: 120; running: smokeMode && experimentSmoke && !workerSmoke; repeat: true
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
    // Commands acknowledge actual control/layout state on a later event pass.
    // Julia keeps servicing the child and owned GLFW screen between passes.
    Timer { interval: 60; running: smokeMode && workerSmoke; repeat: true
        onTriggered: {
            var command = root.uiModel.workerCommand;
            if (command.length === 0) return;
            if (root.workerIssuedCommand !== command) {
                root.workerIssuedCommand = command; root.workerCommandStep = 0;
            }
            if (command === "control") {
                savedControlsSection.currentIndex = 2;
                if (root.workerCommandStep === 0) root.workerCommandStep = 1;
                else if (root.workerCommandStep === 1) {
                    root.workerCommandStep = 2;
                    shell.grabToImage(function(result) {
                        Julia.worker_capture_ack(result.saveToFile(sidebarCapturePrefix+"-active.png"));
                        Julia.worker_control_ack(command);
                    });
                }
            } else if (command === "native_close") {
                if (root.workerCommandStep === 0) {
                    root.workerCommandStep = 1; root.transitionPending = true; Julia.simulate_glfw_close();
                } else if (!root.transitionPending && !root.viewportOpen) Julia.worker_control_ack(command);
            } else if (command.indexOf("close_") === 0 || command.indexOf("reopen_") === 0) {
                var open = command.indexOf("reopen_") === 0;
                if (root.workerCommandStep === 0) { root.workerCommandStep = 1; root.viewportOpen = open; }
                else if (!root.transitionPending && root.viewportLoaded === open) Julia.worker_control_ack(command);
            } else if (command === "capture_small" || command === "capture_large") {
                var small = command === "capture_small";
                if (root.workerCommandStep === 0) {
                    root.workerCommandStep = 1; root.width = small ? 900 : 1100; root.height = small ? 600 : 800;
                    savedControlsSection.currentIndex = 2;
                } else if (root.workerCommandStep === 1) {
                    root.workerCommandStep = 2;
                    sidebarScroll.contentItem.contentY = Math.max(0,sidebarScroll.contentHeight-sidebarScroll.availableHeight);
                    recipeScroll.contentItem.contentY = Math.max(0,recipeScroll.contentHeight-recipeScroll.availableHeight);
                } else if (root.workerCommandStep === 2) {
                    root.workerCommandStep = 3;
                    root.sidebarProbe(small ? "sidebar_probe_small" : "sidebar_probe_large");
                    shell.grabToImage(function(result) {
                        Julia.record_sidebar_capture(small ? "small" : "large", result.saveToFile(sidebarCapturePrefix+(small ? "-small.png" : "-large.png")));
                    });
                }
            } else if (command === "finish") {
                root.lifecycleStep = 12; Julia.lifecycle_complete(); stop();
            }
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
    Timer {
        interval: 350; running: applicationMode && root.uiModel.applicationCaptureRequested; repeat: false
        onTriggered: {
            if (!shell.grabToImage(function(result) {
                if (!result.saveToFile(capturePath)) Julia.smoke_failed("Application control capture could not be saved");
                else Julia.record_capture(true);
            })) Julia.smoke_failed("Application control capture could not be started");
        }
    }
    Timer { interval: workerSmoke ? 350000 : 220000; running: smokeMode; repeat: false
        onTriggered: Julia.smoke_failed("software lifecycle smoke timed out awaiting transition/completion") }
}
