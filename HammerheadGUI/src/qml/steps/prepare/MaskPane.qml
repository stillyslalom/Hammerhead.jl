// Mask sub-page: polygons drawn on the viewer exclude regions without valid
// particles (walls, reflections, shadows); holes restore areas inside them.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

ColumnLayout {
    spacing: 10

    Label {
        text: "Click the image to add vertices and right-click to close the polygon " +
              "(Backspace undoes a vertex, Escape cancels). Click inside a polygon to " +
              "select it; Delete removes it. The shaded area is excluded from the analysis."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }
    Flow {
        Layout.fillWidth: true
        spacing: 8
        enabled: app.hasFrameSize
        Button {
            text: "New hole"
            enabled: !app.maskDrawing
            onClicked: Julia.hh_mask_action("hole")
            ToolTip.visible: hovered
            ToolTip.text: "The next polygon restores an area inside an excluded region"
        }
        Button { text: "Undo vertex"; enabled: app.maskDrawing; onClicked: Julia.hh_mask_action("undo") }
        Button { text: "Close polygon"; enabled: app.maskDrawing; onClicked: Julia.hh_mask_action("close") }
        Button { text: "Delete selected"; enabled: app.maskSelected; onClicked: Julia.hh_mask_action("delete") }
        Button { text: "Clear mask"; enabled: app.hasMask || app.maskDrawing; onClicked: Julia.hh_mask_action("clear") }
    }
    RowLayout {
        spacing: 8
        enabled: app.hasMask
        Label { text: "Margin" }
        SpinBox {
            id: marginBox
            from: 1; to: 64; value: 2; editable: true
            Layout.preferredWidth: 100
        }
        Label { text: "px" }
        Button {
            text: "Grow"
            onClicked: Julia.hh_mask_morph("grow", marginBox.value)
            ToolTip.visible: hovered
            ToolTip.text: "Widen the excluded area (polygons become a fixed raster mask)"
        }
        Button {
            text: "Shrink"
            onClicked: Julia.hh_mask_morph("shrink", marginBox.value)
            ToolTip.visible: hovered
            ToolTip.text: "Narrow the excluded area (polygons become a fixed raster mask)"
        }
    }
    Label {
        text: app.maskStatus
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    RowLayout {
        spacing: 8
        Layout.topMargin: 4
        Button { text: "Open mask image…"; enabled: app.hasFrameSize; onClicked: openMaskDialog.open() }
        Button { text: "Save mask image…"; enabled: app.hasMask; onClicked: saveMaskDialog.open() }
    }

    FileDialog {
        id: openMaskDialog
        title: "Open mask image (white = excluded)"
        nameFilters: ["Images (*.png *.tif *.tiff *.bmp)", "All files (*)"]
        onAccepted: Julia.hh_load_mask(selectedFile.toString())
    }
    FileDialog {
        id: saveMaskDialog
        title: "Save mask image"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "png"
        nameFilters: ["PNG image (*.png)"]
        onAccepted: Julia.hh_save_mask(selectedFile.toString())
    }
}
