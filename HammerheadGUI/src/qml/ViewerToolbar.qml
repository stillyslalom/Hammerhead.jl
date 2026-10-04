// Viewer toolbar along the canvas's right edge: the view mode (edit, zoom,
// pan), reset, and saving the view as an image. Tooltips name the mouse and
// keyboard shortcuts that do the same without the toolbar.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

Pane {
    id: bar
    signal saveView()
    implicitWidth: 76
    width: 76
    padding: 4
    background: Rectangle { color: bar.palette.window }

    ColumnLayout {
        anchors.fill: parent
        spacing: 2
        ButtonGroup { id: modeGroup }
        Repeater {
            model: [{ key: "edit", text: "Edit",
                      tip: "Clicks work on the open step (probe, mask, region, scale, profile…). " +
                           "Also: left-drag zooms to a box, right-drag pans, the wheel zooms, " +
                           "ctrl+click resets." },
                    { key: "zoom", text: "Zoom",
                      tip: "Drag a box to zoom in; clicks do not edit. The wheel also zooms." },
                    { key: "pan", text: "Pan",
                      tip: "Drag to move the view (in Edit mode: right-drag)." }]
            ToolButton {
                text: modelData.text
                checkable: true
                checked: app.viewMode === modelData.key
                ButtonGroup.group: modeGroup
                Layout.fillWidth: true
                onClicked: Julia.hh_set_view_mode(modelData.key)
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: modelData.tip
            }
        }
        ToolSeparator { orientation: Qt.Horizontal; Layout.fillWidth: true }
        ToolButton {
            text: "Reset"
            Layout.fillWidth: true
            onClicked: Julia.hh_reset_view()
            ToolTip.visible: hovered
            ToolTip.delay: 400
            ToolTip.text: "Show the whole image (also ctrl+click on the viewer)"
        }
        ToolButton {
            text: "Save…"
            Layout.fillWidth: true
            onClicked: bar.saveView()
            ToolTip.visible: hovered
            ToolTip.delay: 400
            ToolTip.text: "Save the viewer as a PNG image"
        }
        Item { Layout.fillHeight: true }
    }
}
