import styles from './LectionPage.module.scss'
import Button from '../../atoms/Button/Button.tsx'
import ChatWindow from '../../organisms/ChatWindow/ChatWindow.tsx'
import SecondaryHeader from '../../organisms/SecondaryHeader/SecondaryHeader.tsx'
import { useNavigate } from 'react-router-dom'

const LectionPage = () => {
    const navigate = useNavigate()
    return (
        <div className={styles.wrapper}>
            <SecondaryHeader authorName="Алексей Волков" backUrl="/" />
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
                    <div className={styles.button}>
                        <Button>Начать задание</Button>
                    </div>
                    <div className={styles.chatSector}>
                        <ChatWindow />
                    </div>

                    <div className={styles.buttonSection}>
                        <Button
                            type="secondary"
                            onClick={() => navigate('/results')}
                        >
                            Завершить попытку
                        </Button>
                        <Button onClick={() => navigate('/results')}>
                            Посмотреть результат
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default LectionPage
