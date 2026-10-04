import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

StepPage {
    title: "Test pair"
    guidance: app.mode === "ensemble"
        ? "Run the current settings on the first pairs as an ensemble, exactly as the batch will."
        : app.mode === "tracking"
        ? "Track particles through up to ten frames from the representative pair, with the " +
          "settings the batch will use. The viewer draws the tracks."
        : app.mode === "ptv"
        ? "Match the particles of the representative pair, exactly as the batch will. Valid " +
          "matches show in blue, flagged matches in red."
        : "Run the current settings on the representative pair, exactly as the batch will. " +
          "Valid vectors show in blue, flagged vectors in red."

    RowLayout {
        spacing: 12
        Button {
            text: app.mode === "ensemble" ? "Test ensemble" :
                  app.mode === "tracking" ? "Test tracking from pair " + app.pairIndex :
                  "Test pair " + app.pairIndex
            highlighted: true
            enabled: !app.testRunning && app.analysisProblem === ""
            onClicked: Julia.hh_test()
        }
        BusyIndicator {
            running: app.testRunning
            visible: running
            implicitWidth: 32
            implicitHeight: 32
        }
    }
    Label {
        text: app.analysisProblem
        visible: text !== "" && !app.testRunning
        color: "#b06f00"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    Label {
        text: app.testStatus
        visible: text !== ""
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    Frame {
        visible: app.testLines !== ""
        Layout.fillWidth: true
        Layout.topMargin: 4
        ColumnLayout {
            anchors.left: parent.left
            anchors.right: parent.right
            spacing: 6
            Label {
                text: "Settings or pair changed since this test (pair " + app.testPair + ") — test again"
                visible: app.testStale
                color: "#b06f00"
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
            Label {
                text: app.testLines
                lineHeight: 1.25
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }
}
