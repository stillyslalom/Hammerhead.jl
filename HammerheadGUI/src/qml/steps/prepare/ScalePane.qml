// Scale sub-page: pixel size and frame interval attached to the results
// (display metadata; vectors are computed in pixels), typed in or measured
// from two points of known separation on the viewer. In a stereo window
// (`stereo`) lengths are the calibration's world units, so only the frame
// interval and the unit names are set.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

ColumnLayout {
    property bool stereo: false
    spacing: 10

    Label {
        text: stereo
            ? "Stereo vectors are in the calibration's world units (" + app.scaleWorldUnit +
              ") per frame. Enter the time between exposures to show velocities."
            : "Attach a physical scale so results show positions and velocities in " +
              "physical units. Without one they stay in pixels and frames."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }
    GridLayout {
        columns: 3
        columnSpacing: 8
        rowSpacing: 6

        Label { text: "Pixel size"; visible: !stereo }
        TextField {
            visible: !stereo
            text: app.scalePixelSize
            placeholderText: "e.g. 0.02"
            Layout.preferredWidth: 110
            onEditingFinished: if (text !== app.scalePixelSize) Julia.hh_set_scale("pixel_size", text)
        }
        RowLayout {
            visible: !stereo
            TextField {
                text: app.scaleLengthUnit
                placeholderText: "mm"
                Layout.preferredWidth: 70
                onEditingFinished: if (text !== app.scaleLengthUnit) Julia.hh_set_scale("length_unit", text)
            }
            Label { text: "per pixel"; opacity: 0.75 }
        }

        Label { text: "Time between frames" }
        TextField {
            text: app.scaleDt
            placeholderText: "e.g. 0.001"
            Layout.preferredWidth: 110
            onEditingFinished: if (text !== app.scaleDt) Julia.hh_set_scale("dt", text)
        }
        TextField {
            text: app.scaleTimeUnit
            placeholderText: "s"
            Layout.preferredWidth: 70
            onEditingFinished: if (text !== app.scaleTimeUnit) Julia.hh_set_scale("time_unit", text)
        }

        Label { text: "Length unit"; visible: stereo }
        TextField {
            visible: stereo
            text: app.scaleLengthUnit
            placeholderText: stereo ? app.scaleWorldUnit : ""
            Layout.preferredWidth: 110
            onEditingFinished: if (text !== app.scaleLengthUnit) Julia.hh_set_scale("length_unit", text)
        }
        Label { visible: stereo; text: "" }
    }
    RowLayout {
        Label { text: app.scaleSummary; wrapMode: Text.WordWrap; Layout.fillWidth: true }
        Button { text: "Remove scale"; enabled: app.hasScale; onClicked: Julia.hh_clear_scale() }
    }

    Label { visible: !stereo; text: "Measure the pixel size"; font.weight: Font.DemiBold; Layout.topMargin: 8 }
    Label {
        visible: !stereo
        text: "Click two points of known separation on the image (a ruler or calibration " +
              "target), then enter their distance; the pixel size follows. A ruler " +
              "photographed separately at the frames' magnification can be loaded instead."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }
    RowLayout {
        visible: !stereo
        spacing: 8
        Button { text: "Load a ruler image…"; onClicked: rulerDialog.open() }
        Button {
            text: "Use the frames"
            enabled: app.rulerName !== ""
            onClicked: Julia.hh_clear_ruler()
        }
        Label {
            text: app.rulerName !== "" ? "measuring on " + app.rulerName : "measuring on the frames"
            opacity: 0.75
            elide: Text.ElideMiddle
            Layout.fillWidth: true
        }
    }
    RowLayout {
        visible: !stereo
        spacing: 8
        enabled: app.hasFrameSize
        Label { text: "Distance" }
        TextField {
            text: app.scaleSeparation
            Layout.preferredWidth: 110
            onEditingFinished: if (text !== app.scaleSeparation) Julia.hh_set_scale("separation", text)
        }
        Label { text: app.scaleMeasureUnit }
        Button { text: "Clear points"; onClicked: Julia.hh_clear_scale_points() }
    }
    Label {
        visible: !stereo
        text: app.scaleMeasure
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    Label {
        text: app.scaleError
        visible: text !== ""
        color: "#c42b1c"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    FileDialog {
        id: rulerDialog
        title: "Ruler image"
        nameFilters: ["Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg)", "All files (*)"]
        onAccepted: Julia.hh_load_ruler(selectedFile.toString())
    }
}
