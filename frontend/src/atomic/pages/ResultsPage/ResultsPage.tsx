import styles from './ResultsPage.module.scss'
import SecondaryHeader from '../../organisms/SecondaryHeader/SecondaryHeader.tsx'
import Button from '../../atoms/Button/Button.tsx'
import ChatWindow from '../../organisms/ChatWindow/ChatWindow.tsx'
import { useNavigate } from 'react-router-dom'
import CommentsSection from '../../organisms/CommentsSection/CommentsSection.tsx'
import BarChart from '../../organisms/BarChart/BarChart.tsx'
import DetailsComments from '../../organisms/DetailsComments/DetailsComments.tsx'

const answer1 =
    '«Добрый день! Я продавец этого магазина, могу ли я вам как-то помочь?»'
const toDevelop1 =
    '«Здравствуйте! Меня зовут Егор, я продавец этого магазина, могу ли я вам как-то помочь?»'
const ResultsPage = () => {
    const navigate = useNavigate()
    return (
        <div className={styles.wrapper}>
            <SecondaryHeader authorName="Алексей Волков" backUrl="/lection" />
            <div className={styles.content}>
                <div className={styles.situationSector}>
                    <div className={styles.title}>Ситуация</div>
                    <div className={styles.description}>
                        Вы работаете консультантом в автосалоне премиум-класса.
                        Сегодня в салон зашёл потенциальный покупатель, но вы
                        пока не знаете, кто он и какие автомобили рассматривает.
                        Ваша задача — определить его потребности и интересы,
                        чтобы предложить подходящий автомобиль и успешно
                        завершить продажу.
                    </div>

                    <div className={styles.chatSector}>
                        <ChatWindow />
                    </div>
                    <div className={styles.score}>
                        <h2 className={styles.score_header}>Ваш результат</h2>
                        <div className={styles.score_info}>
                            <div className={styles.score_mark}>5/10</div>
                            <div className={styles.score_feedback}>
                                Есть куда стремится. <br /> Изучите материал еще раз
                            </div>
                            <div className={styles.score_button}>
                                <Button type="secondary">
                                    Смотреть подробнее
                                </Button>
                            </div>
                        </div>
                    </div>
                    <BarChart />
                    <DetailsComments
                        answer={answer1}
                        toDevelop={toDevelop1}
                        hint={'стр 4 , пункт 2.1'}
                        numberOfPoint={1}
                    />
                    <div className={styles.commentsSector}>
                        <CommentsSection />
                    </div>

                    <div className={styles.buttonSection}>
                        <Button type="secondary">Оспорить оценку</Button>
                        <Button onClick={() => navigate('/')}>
                            Вернуться к курсу
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ResultsPage
