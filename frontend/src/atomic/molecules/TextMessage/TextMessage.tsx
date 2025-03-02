import React from 'react'
import styles from './TextMessage.module.scss'

interface TextMessageProps {
    text: string
}
const TextMessage: React.FC<TextMessageProps> = ({ text }) => {
    return (
        <div className={styles.wrapper}>
            <div className={styles.text}>{text}</div>
        </div>
    )
}

export default TextMessage
