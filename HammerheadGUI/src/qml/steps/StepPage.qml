// Common frame for a step page: title, short guidance, then the page content
// (children go into the column), scrollable when the window is short. A page
// may add a `footer`: controls pinned below the scrolling content.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: page
    property string title: ""
    property string guidance: ""
    default property alias content: column.data
    property alias footer: footerColumn.data

    ScrollView {
        id: scroll
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: footerPane.visible ? footerPane.top : parent.bottom
        contentWidth: availableWidth
        contentHeight: column.implicitHeight + 32
        clip: true

        ColumnLayout {
            id: column
            width: scroll.availableWidth - 40
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

    Rectangle {
        visible: footerPane.visible
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: footerPane.top
        height: 1
        color: palette.mid
    }
    Pane {
        id: footerPane
        visible: footerColumn.children.length > 0
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        leftPadding: 20
        rightPadding: 20
        topPadding: 10
        bottomPadding: 12

        ColumnLayout {
            id: footerColumn
            anchors.left: parent.left
            anchors.right: parent.right
            spacing: 6
        }
    }
}
