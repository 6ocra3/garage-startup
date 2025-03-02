import { useState } from 'react'
import styles from './ChatWindow.module.scss'
import TextMessage from '../../molecules/TextMessage/TextMessage.tsx'
import ChatInput from '../../molecules/ChatInput/ChatInput.tsx'
import classNames from 'classnames'

export interface Message {
    type: 'text' | 'audio'
    content: string
    isYou: boolean
}

const ChatWindow = () => {
    const [messages, setMessages] = useState<Message[]>([])

    const handlePlayAudio = async (url: string) => {
        try {
            const response = await fetch(url)
            const blob = await response.blob()
            const audioUrl = URL.createObjectURL(blob)
            const audio = new Audio(audioUrl)
            await audio.play()
        } catch (error) {
            console.error('Ошибка воспроизведения аудио:', error)
        }
    }

    return (
        <div className={styles.chatContainer}>
            <div className={styles.chatCard}>
                <div className={styles.chatContent}>
                    {messages.map((msg, index) => (
                        <div
                            key={index}
                            className={classNames(
                                styles.message,
                                msg.isYou && styles.isYourMessage
                            )}
                        >
                            {msg.type === 'text' ? (
                                <TextMessage text={msg.content} />
                            ) : (
                                <div>
                                    <p className={styles.owner}>
                                        {msg.isYou ? 'Вы' : 'Клиент'}
                                    </p>
                                    <div className={styles.audio}>
                                        <button
                                            className={styles.playButton}
                                            onClick={() =>
                                                handlePlayAudio(msg.content)
                                            }
                                        >
                                            <img
                                                src="/play-button.svg"
                                                alt="audio"
                                            />
                                        </button>
                                        {msg.isYou ? (
                                            <img
                                                src="/active-audio.svg"
                                                alt="audio"
                                            />
                                        ) : (
                                            <img
                                                src="/another-audio.svg"
                                                alt="audio"
                                            />
                                        )}

                                        <img
                                            className={styles.descriptor}
                                            src="/descriptor.svg"
                                            alt="audio"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
                <ChatInput setMessages={setMessages} messages={messages} />
            </div>
        </div>
    )
}

export default ChatWindow
