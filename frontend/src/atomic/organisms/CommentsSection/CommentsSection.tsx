import styles from './CommentsSection.module.scss'
import Box from '../../atoms/Box/Box.tsx'

const positiveArray = [
    'Предложенная машина заинтересовала покупателя',
    'Инициативность',
    'Учтены пожелания покупателя',
]
const negativeArray = [
    'Не было конкретики про характеристики машины',
    'Изначально не учли ценовой диапозон',
    'Покупателю не был предложен тест-драйв',
]
const CommentsSection = () => {
    return (
        <div className={styles.wrapper}>
            <div className={styles.title}>Комментарии</div>
            <div className={styles.content}>
                <Box
                    title="Хорошие моменты"
                    comments={positiveArray}
                    backgroundColor="#F9FFFA"
                />
                <Box
                    title="Спорные моменты"
                    comments={negativeArray}
                    backgroundColor="#FFF7F7"
                />
            </div>
        </div>
    )
}

export default CommentsSection
