import React, { useState } from 'react'
import styles from './ChatInput.module.scss'
import { Message } from '../../organisms/ChatWindow/ChatWindow.tsx'
import { useReactMediaRecorder } from 'react-media-recorder'

interface ChatInputProps {
    setMessages: (msg: Message[]) => void
    messages: Message[]
}

const ChatInput: React.FC<ChatInputProps> = ({ setMessages, messages }) => {
    const [text, setText] = useState<string>('')
    const [hasAudio, setHasAudio] = useState(false);

    const { startRecording, stopRecording, mediaBlobUrl, status } =
        useReactMediaRecorder({
            audio: true,
        })

    const handleClick = () => {
        if(status != "recording"){
            startRecording();
        } else {
            setHasAudio(true)
            stopRecording();
        }
    }

    const sendMessage = (msg: string, type: 'text' | 'audio' = 'text', isYou = true) => {
        if (isYou) {
            setMessages([...messages, { type, content: msg, isYou: true }])
        } else {
            setMessages([...messages, { type, content: msg, isYou: true }, { type, content: msg, isYou: false }])
        }

        setText('')
    }

    const handleSendVoice = async () => {
        if (mediaBlobUrl) {
            try {
                // Загружаем Blob из mediaBlobUrl
                const response = await fetch(mediaBlobUrl);
                const blob = await response.blob();
                const blobUrl = URL.createObjectURL(blob);
                sendMessage(blobUrl, 'audio');
                // Отправляем Blob на бэкенд (убедитесь, что URL корректный, например, если сервер на другом домене – настройте CORS)
                await fetch("http://51.250.34.157:8000/message", {
                    method: "POST",
                    headers: {
                        "Content-Type": "audio/webm"
                    },
                    body: blob,
                })  .then(response => response.blob())
                    .then(blob => {
                        const audioUrl = URL.createObjectURL(blob);
                        const audioElement = new Audio(audioUrl);
                        audioElement.controls = true;
                        document.body.appendChild(audioElement);
                        audioElement.play();
                    })
                    .catch(error => console.error('Ошибка:', error));

                sendMessage(blobUrl, 'audio', false);
                setHasAudio(false);
            } catch (error) {
                console.error('Ошибка при загрузке или отправке аудио Blob:', error);
            }
        }
    };

    return (
        <div className={styles.inputContainer}>
            {hasAudio ? (
                <button className={styles.clip}>
                    <img src="/play-button.svg" alt="audio" />
                </button>
            ) : (
                <button className={styles.clip}>
                    <svg
                        width="32"
                        height="32"
                        viewBox="0 0 32 32"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M6.66675 9.328L6.66675 20C6.66675 22.4507 7.64942 24.7813 9.43475 26.5653C11.2201 28.3493 13.5507 29.3333 16.0001 29.3333C18.4494 29.3333 20.7814 28.3507 22.5654 26.5653C24.3494 24.78 25.3334 22.4507 25.3334 20L25.3334 9.33333L22.6667 9.33333L22.6667 20C22.6667 21.7373 21.9614 23.4 20.6801 24.68C19.4001 25.9613 17.7374 26.6667 16.0001 26.6667C14.2627 26.6667 12.6014 25.9613 11.3201 24.68C10.0387 23.4 9.33341 21.7373 9.33341 20L9.33341 9.33333L9.33341 9.328C9.33341 8.27466 9.75208 7.276 10.5147 6.51599C11.2761 5.75466 12.2774 5.33466 13.3361 5.33333C13.8597 5.33406 14.3779 5.4386 14.8608 5.64092C15.3437 5.84323 15.7817 6.1393 16.1494 6.512C16.9024 7.25837 17.3282 8.27314 17.3334 9.33333L17.3334 20C17.3277 20.3508 17.184 20.6851 16.9334 20.9307C16.6877 21.1829 16.3522 21.3276 16.0001 21.3333C15.6641 21.3333 15.3241 21.1867 15.0694 20.932C14.8175 20.6866 14.6728 20.3516 14.6667 20L14.6667 9.33333L12.0001 9.33333L12.0001 20C12.0001 21.0533 12.4201 22.0547 13.1854 22.8187C13.9481 23.5813 14.9481 24 16.0001 24C17.0521 24 18.0534 23.5813 18.8187 22.816C19.1922 22.4482 19.489 22.0098 19.6918 21.5264C19.8946 21.0431 19.9994 20.5242 20.0001 20L20.0001 9.332C19.9995 8.45709 19.8256 7.59096 19.4885 6.78359C19.1514 5.97622 18.6578 5.2436 18.0361 4.628C17.4203 4.00698 16.6876 3.51408 15.8803 3.17775C15.073 2.84142 14.207 2.66831 13.3325 2.66844C12.4579 2.66856 11.592 2.84191 10.7848 3.17847C9.97759 3.51503 9.24503 4.00814 8.62941 4.62933C8.00741 5.24352 7.51361 5.97516 7.17669 6.78176C6.83977 7.58836 6.66643 8.45386 6.66675 9.328Z"
                            fill="#DEDEDE"
                        />
                    </svg>
                </button>
            )}
            {hasAudio ? (
                <div className={styles.shortAudio}>
                    <img src="/short-audio.svg" alt="audio" />
                </div>
            ) : (
                <input
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Запишите свой ответ"
                    className={styles.input}
                />
            )}

            {/*<button*/}
            {/*    onClick={() => sendMessage(text)}*/}
            {/*    className={styles.sendButton}*/}
            {/*>*/}
            {/*    <Send size={20} />*/}
            {/*</button>*/}
            {!hasAudio && (
                <button onClick={handleClick} className={styles.micButton}>
                    {status !== 'recording' && (
                        <svg
                            width="32"
                            height="32"
                            viewBox="0 0 32 32"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                d="M21.3334 16V8C21.3334 5.044 18.9534 2.63867 16.0281 2.63867C15.9343 2.63995 15.8409 2.65112 15.7494 2.67201C14.3804 2.73838 13.0893 3.32848 12.1433 4.32021C11.1972 5.31194 10.6686 6.6294 10.6667 8V16C10.6667 18.9413 13.0587 21.3333 16.0001 21.3333C18.9414 21.3333 21.3334 18.9413 21.3334 16ZM13.3334 16V8C13.3334 6.52934 14.5294 5.33334 16.0001 5.33334C16.073 5.33262 16.1456 5.32593 16.2174 5.31334C17.5841 5.41334 18.6667 6.58 18.6667 8V16C18.6667 17.4707 17.4707 18.6667 16.0001 18.6667C14.5294 18.6667 13.3334 17.4707 13.3334 16Z"
                                fill="#4760F0"
                            />
                            <path
                                d="M7.99992 16H5.33325C5.33325 21.4293 9.41459 25.9147 14.6666 26.5747V29.3333H17.3333V26.5747C22.5853 25.9147 26.6666 21.4307 26.6666 16H23.9999C23.9999 20.412 20.4119 24 15.9999 24C11.5879 24 7.99992 20.412 7.99992 16Z"
                                fill="#4760F0"
                            />
                        </svg>
                    )}

                    {status === 'recording' && (
                        <svg
                            width="32"
                            height="32"
                            viewBox="0 0 32 32"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                fill-rule="evenodd"
                                clip-rule="evenodd"
                                d="M5.33337 7.99992C5.33337 7.29267 5.61433 6.6144 6.11442 6.1143C6.61452 5.6142 7.2928 5.33325 8.00004 5.33325H24C24.7073 5.33325 25.3856 5.6142 25.8857 6.1143C26.3858 6.6144 26.6667 7.29267 26.6667 7.99992V23.9999C26.6667 24.7072 26.3858 25.3854 25.8857 25.8855C25.3856 26.3856 24.7073 26.6666 24 26.6666H8.00004C7.2928 26.6666 6.61452 26.3856 6.11442 25.8855C5.61433 25.3854 5.33337 24.7072 5.33337 23.9999V7.99992ZM24 7.99992H8.00004V23.9999H24V7.99992Z"
                                fill="#4760F0"
                            />
                        </svg>
                    )}
                </button>
            )}
            {hasAudio && (
                <>
                    <button
                        onClick={() => {
                            setHasAudio(false)
                        }}
                        className={styles.sendButton}
                    >
                        <img
                            // className={}
                            src="/trash.svg"
                            alt="audio"
                        />
                    </button>
                    <div className={styles.trash}></div>
                    <button
                        onClick={handleSendVoice}
                        className={styles.sendButton}
                    >
                        <svg
                            width="32"
                            height="32"
                            viewBox="0 0 32 32"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                fill-rule="evenodd"
                                clip-rule="evenodd"
                                d="M23.988 8.01337L7.19872 14.084L12.792 17.3214L17.724 12.388C17.9742 12.138 18.3135 11.9976 18.6672 11.9978C19.0209 11.9979 19.36 12.1385 19.61 12.3887C19.8601 12.6389 20.0004 12.9781 20.0003 13.3318C20.0002 13.6855 19.8596 14.0247 19.6094 14.2747L14.676 19.208L17.916 24.8L23.988 8.01337ZM24.4187 5.02137C26.012 4.44404 27.556 5.98804 26.9787 7.58137L19.936 27.0547C19.3574 28.652 17.176 28.8467 16.324 27.376L12.0347 19.9654L4.62405 15.676C3.15338 14.824 3.34805 12.6427 4.94538 12.064L24.4187 5.02137Z"
                                fill="#4760F0"
                            />
                        </svg>
                    </button>
                </>
            )}
        </div>
    )
}

export default ChatInput
