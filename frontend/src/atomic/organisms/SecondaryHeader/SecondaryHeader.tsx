import React from 'react'
import styles from './SecondaryHeader.module.scss'
import { useNavigate } from 'react-router-dom'
import Button from '../../atoms/Button/Button.tsx'
import BackSvg from '/back.svg'

interface SecondaryHeaderProps {
    authorName: string
    backUrl: string
}
const SecondaryHeader: React.FC<SecondaryHeaderProps> = ({
    authorName,
    backUrl,
}) => {
    const navigate = useNavigate()
    return (
        <header className={styles.wrapper}>
            <div className={styles.content}>
                <div className={styles.contentLeft}>
                    <Button type="secondary" onClick={() => navigate(backUrl)}>
                        <img src={BackSvg} alt="back" />
                        Вернуться назад
                    </Button>
                </div>
                <div className={styles.contentRight}>
                    <div className={styles.mainText}>{authorName}</div>
                    <div className={styles.text}>автор задания</div>
                </div>
            </div>
        </header>
    )
}

export default SecondaryHeader
