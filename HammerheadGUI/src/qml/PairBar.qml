// Bottom bar: choose the representative pair and which frame the canvas shows
// (in a stereo window also which camera).
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

ToolBar {
    property bool stereo: false

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 12
        anchors.rightMargin: 12
        spacing: 8
        enabled: app.pairCount > 0 || stereo  // the camera switch also serves Calibration

        Label { text: "Pair" }
        ToolButton {
            text: "‹"
            enabled: app.pairIndex > 1
            onClicked: Julia.hh_select_pair(app.pairIndex - 1)
        }
        Slider {
            Layout.preferredWidth: 240
            from: 1
            to: Math.max(2, app.pairCount)
            stepSize: 1
            snapMode: Slider.SnapAlways
            value: app.pairIndex
            enabled: app.pairCount > 1
            onMoved: Julia.hh_select_pair(Math.round(value))
        }
        ToolButton {
            text: "›"
            enabled: app.pairIndex < app.pairCount
            onClicked: Julia.hh_select_pair(app.pairIndex + 1)
        }
        Label {
            text: app.pairCount > 0 ? app.pairIndex + " of " + app.pairCount : "no pairs"
            Layout.preferredWidth: 90
        }
        ToolSeparator {}
        ButtonGroup { id: frameGroup }
        Button {
            text: "Frame A"
            flat: true
            checkable: true
            checked: app.shown === "a"
            ButtonGroup.group: frameGroup
            onClicked: Julia.hh_show_frame("a")
        }
        Button {
            text: "Frame B"
            flat: true
            checkable: true
            checked: app.shown === "b"
            ButtonGroup.group: frameGroup
            onClicked: Julia.hh_show_frame("b")
        }
        ToolSeparator { visible: stereo }
        ButtonGroup { id: cameraGroup }
        Repeater {
            model: stereo ? [1, 2] : []
            Button {
                text: "Camera " + modelData
                flat: true
                checkable: true
                checked: stereo && app.camera === modelData
                ButtonGroup.group: cameraGroup
                onClicked: Julia.hh_set_camera(modelData)
            }
        }
        Item { Layout.fillWidth: true }
        Label {
            text: app.framesSummary
            elide: Text.ElideRight
            Layout.maximumWidth: 360
            opacity: 0.75
        }
    }
}
