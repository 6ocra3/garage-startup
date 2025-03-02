import React from 'react'
import styles from './MainPage.module.scss'
import Header from '../../organisms/Header/Header.tsx'
import ReactComponent from '/authorAvatar.svg'
import Button from '../../atoms/Button/Button.tsx'
import { useNavigate } from 'react-router-dom'

const MainPage: React.FC = () => {
    const navigate = useNavigate()
    return (
        <div className={styles.wrapper}>
            <Header
                durationInMinutes={8}
                authorName="Алексей Волков"
                AuthorAvatar={ReactComponent}
            />
            <div className={styles.content}>
                <div className={styles.mainContent}>
                    <iframe
                        src="/pdf-example-document.pdf"
                        width="100%"
                        height="100%"
                    />
                </div>
                <div className={styles.buttonSection}>
                    <Button type="secondary">
                        <img
                            src="/download-docs.svg"
                        />
                        Скачать файл
                    </Button>
                    <Button onClick={() => navigate('/lection')}>
                        Перейти к заданию
                        <img
                            src="/go-to-ans.svg"
                        />
                    </Button>
                </div>
            </div>
        </div>
    )
}

export default MainPage
