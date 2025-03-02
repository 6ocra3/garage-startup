import React from 'react'
import styles from './Header.module.scss'

interface HeaderProps {
    authorName: string
    AuthorAvatar: string
    durationInMinutes: number
}
const Header:React.FC<HeaderProps> = ({authorName, AuthorAvatar, durationInMinutes}) => {
    return (
        <header className={styles.wrapper}>
            <div className={styles.content}>
                <div className={styles.contentLeft}>
                    <img alt='avatar' src={AuthorAvatar}/>
                    <div className={styles.authorInfo}>
                        <div className={styles.mainText}>{authorName}</div>
                        <div className={styles.text}>автор лекции</div>
                    </div>
                </div>
                <div className={styles.contentRight}>
                    <div className={styles.boldText}>Лекция</div>
                    <div className={styles.text}>читается за {durationInMinutes} минут</div>
                </div>
            </div>
        </header>
    )
}

export default Header