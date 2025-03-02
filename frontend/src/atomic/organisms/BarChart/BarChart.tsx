import styles from './BarChart.module.scss'

const BarChart = () => {
    const barW = 40
    const gap = 24
    const values = [50, 75, 40, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return (
        <div className={styles.chart}>
            {values.map((val, index) => {
                const x = (barW + gap) * index
                const height = val
                return (
                    <div style={{ left: x }} className={styles.section}>
                        <div className={styles.background}></div>
                        <div
                            style={{ height: height + '%' }}
                            className={styles.bar}
                        ></div>
                    </div>
                )
            })}
        </div>
    )
}

export default BarChart
