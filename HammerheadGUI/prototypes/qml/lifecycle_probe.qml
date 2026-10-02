import QtQuick
import QtQuick.Controls
import Makie
import jlqml

ApplicationWindow {
    id: root
    visible: scenario !== "construction"
    width: 760; height: 530
    title: "Offscreen Qt lifecycle child"
    // Loader has a model context property of its own; bind through the root.
    property var dataModel: model
    property bool viewportActive: true
    property bool separate: false
    property int phase: 1
    property int completedCycles: 0
    property int resizeStep: 0
    property bool captured: false
    property bool geometryLogged: false
    property bool releaseFramePending: false
    Loader { id: integrated; anchors.fill: parent
        active: root.viewportActive && !root.separate; sourceComponent: viewport }
    Window {
        id: separateWindow; visible: root.visible && root.separate && root.viewportActive
        width: 710; height: 490
        Loader { id: external; anchors.fill: parent
            active: root.viewportActive && root.separate; sourceComponent: viewport }
    }
    Component { id: viewport; MakieArea {
        anchors.fill: parent
        scene: root.dataModel.generation === 1 ? initialPlot : root.dataModel.plot; focus: true
        Component.onCompleted: Julia.qml_stage("native_item_created_" + width + "x" + height + "_visible_" + visible)
    } }
    Timer {
        interval: 100; running: true; repeat: true
        onTriggered: {
            if (Julia.sync_lifecycle()) { Qt.exit(2); return; }
            if (root.phase === 0) {
                Julia.create_viewport(); root.phase = 10;
            } else if (root.phase === 10) {
                // Allow the queued JuliaPropertyMap updates to reach Qt before
                // constructing a new native renderer with the next figure.
                root.viewportActive = true; root.phase = 1;
            } else if (root.phase === 1) {
                if (!root.geometryLogged) {
                    root.geometryLogged = true;
                    let current = root.separate ? external.item : integrated.item;
                    Julia.qml_stage("viewport_geometry_" + current.width + "x" + current.height + "_window_visible_" + root.visible);
                }
                // Offscreen windows need an explicit frame request. Waiting for
                // first_frame before grabToImage can wait forever on this platform.
                if (!root.captured && scenario !== "construction") {
                    root.phase = 2;
                    let item = root.separate ? external.item : integrated.item;
                    item.grabToImage(function(result) {
                        Julia.record_capture(result.saveToFile(capturePath));
                        root.captured = true; root.phase = 1;
                    });
                    return;
                }
                if (scenario !== "construction" && model.rendered !== model.generation) {
                    (root.separate ? external.item : integrated.item).update(); return;
                }
                if (scenario === "resize" && root.resizeStep < 4) {
                    root.resizeStep += 1;
                    root.width = root.resizeStep % 2 ? 900 : 760;
                    root.height = root.resizeStep % 2 ? 650 : 530;
                    Julia.qml_stage("resize_requested"); return;
                }
                root.phase = 3;
            } else if (root.phase === 3) {
                if (scenario === "failure") Julia.processing_failure();
                Julia.request_release();
                if (scenario !== "construction") (root.separate ? external.item : integrated.item).update();
                root.phase = 4;
            } else if (root.phase === 4) {
                if (model.released !== model.generation) {
                    if (!root.releaseFramePending) {
                        root.releaseFramePending = true;
                        (root.separate ? external.item : integrated.item).grabToImage(function(result) {
                            Julia.qml_stage("release_frame_requested");
                        });
                    }
                    return;
                }
                root.viewportActive = false;
                Julia.qml_stage("qml_loader_deactivated"); root.phase = 5;
            } else if (root.phase === 5) {
                Julia.detach_viewport(model.generation);
                root.completedCycles += 1;
                let target = (scenario === "reopen" || scenario === "separate" || scenario === "reuse") ? cycles : 1;
                if (root.completedCycles < target) {
                    if (scenario === "separate") root.separate = !root.separate;
                    root.geometryLogged = false; root.releaseFramePending = false;
                    root.phase = 0;
                } else {
                    root.visible = false; separateWindow.visible = false;
                    Julia.lifecycle_complete(); stop(); Qt.exit(0);
                }
            }
        }
    }
    Timer { interval: 45000; running: true; repeat: false
        onTriggered: { Julia.qml_stage("qml_deadline_exceeded"); Qt.exit(3); } }
}
