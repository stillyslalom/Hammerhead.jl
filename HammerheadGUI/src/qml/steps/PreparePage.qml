import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

StepPage {
    title: "Prepare"
    guidance: "Preprocessing, mask, region of interest and physical scale. Settings opened " +
              "from a file are applied as they are; the mask and region show in the viewer."

    Label { text: app.prepareSummary; wrapMode: Text.WordWrap; Layout.fillWidth: true }
    Label {
        text: "Editing them in this window comes in a later update; until then, open " +
              "settings that contain them."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
        Layout.topMargin: 8
    }
}
