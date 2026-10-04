// Common frame for a step page: title, short guidance, then the page content
// (children go into the column), scrollable when the window is short.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: page
    property string title: ""
    property string guidance: ""
    default property alias content: column.data

    contentWidth: availableWidth
    contentHeight: column.implicitHeight + 32
    clip: true

    ColumnLayout {
        id: column
        width: page.availableWidth - 40
        x: 20
        y: 16
        spacing: 12

        Label {
            text: page.title
            font.pixelSize: 22
            font.weight: Font.DemiBold
        }
        Label {
            text: page.guidance
            visible: text !== ""
            wrapMode: Text.WordWrap
            opacity: 0.75
            Layout.fillWidth: true
        }
    }
}
