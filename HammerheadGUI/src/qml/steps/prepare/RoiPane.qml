// Region sub-page: the rectangle of each frame to analyze (inclusive pixel
// bounds), typed in or set with two clicks on the viewer.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

ColumnLayout {
    spacing: 10

    Label {
        text: "Analyze only part of each frame: click two opposite corners on the image, " +
              "or type the bounds (inclusive pixels). Vectors keep full-image coordinates."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }
    GridLayout {
        columns: 4
        columnSpacing: 8
        rowSpacing: 6
        enabled: app.hasFrameSize

        Label { text: "" }
        Label { text: "first"; opacity: 0.75 }
        Label { text: "last"; opacity: 0.75 }
        Label { text: "" }

        Label { text: "Rows (y)" }
        TextField { id: r1; text: app.roiRowFirst; Layout.preferredWidth: 80; onAccepted: apply() }
        TextField { id: r2; text: app.roiRowLast; Layout.preferredWidth: 80; onAccepted: apply() }
        Label { text: "" }

        Label { text: "Columns (x)" }
        TextField { id: c1; text: app.roiColFirst; Layout.preferredWidth: 80; onAccepted: apply() }
        TextField { id: c2; text: app.roiColLast; Layout.preferredWidth: 80; onAccepted: apply() }
        Label { text: "" }
    }
    RowLayout {
        spacing: 8
        enabled: app.hasFrameSize
        Button { text: "Apply bounds"; onClicked: apply() }
        Button { text: "Full image"; enabled: app.hasRoi; onClicked: Julia.hh_clear_roi() }
    }
    Label {
        text: app.roiSummary
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    Label {
        text: app.roiError
        visible: text !== ""
        color: "#c42b1c"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    function apply() { Julia.hh_set_roi(r1.text, r2.text, c1.text, c2.text) }
}
