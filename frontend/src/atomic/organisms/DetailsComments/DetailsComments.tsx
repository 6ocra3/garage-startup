import React from 'react'
import styles from './DetailsComments.module.scss'
import Box from '../../atoms/Box/Box.tsx'
interface DetailsCommentsProps {
    numberOfPoint: number
    answer: string
    toDevelop: string
    hint: string
}

const DetailsComments: React.FC<DetailsCommentsProps> = ({
    numberOfPoint,
    answer,
    toDevelop,
    hint,
}) => {
    return (
        <div className={styles.wrapper}>
            <div className={styles.content}>
                <Box comments={[answer]} title={'Ответ' + numberOfPoint} />
                <Box
                    comments={[toDevelop]}
                    title="Что можно улучшить"
                    hint={hint}
                />
            </div>
        </div>
    )
}

export default DetailsComments
