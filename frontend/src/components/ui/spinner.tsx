import { styled } from 'decorock'

const RectSpinnerContainer = styled.div`
  @keyframes sk-stretchdelay {
    0%,
    40%,
    100% {
      transform: scaleY(0.4);
    }

    20% {
      transform: scaleY(1);
    }
  }

  width: 70px;
  height: 40px;
  margin: 100px auto;
  font-size: 10px;
  text-align: center;

  & > div {
    display: inline-block;
    width: 6px;
    height: 100%;
    margin: 0 4px;
    animation: sk-stretchdelay 1.2s infinite ease-in-out;
    background-color: ${(p) => p.theme.colors.primary};
  }

  div:nth-child(2) {
    animation-delay: -1.1s;
  }

  div:nth-child(3) {
    animation-delay: -1s;
  }

  div:nth-child(4) {
    animation-delay: -0.9s;
  }

  div:last-child {
    animation-delay: -0.8s;
  }
`

export const RectSpinner = () => (
  <RectSpinnerContainer>
    <div class="rect1" />
    <div class="rect2" />
    <div class="rect3" />
    <div class="rect4" />
    <div class="rect5" />
  </RectSpinnerContainer>
)

const CircleSpinnerContainer = styled.div`
  position: relative;
  width: 30px;
  height: 30px;

  & > div {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    &::before {
      display: block;
      width: 15%;
      height: 15%;
      border-radius: 100%;
      margin: 0 auto;
      animation: sk-circleFadeDelay 1.2s infinite ease-in-out both;
      background-color: ${(p) => p.theme.colors.primary};
      content: '';
    }
  }

  div:nth-child(2) {
    transform: rotate(30deg);

    &::before {
      animation-delay: -1.1s;
    }
  }

  div:nth-child(3) {
    transform: rotate(60deg);

    &::before {
      animation-delay: -1s;
    }
  }

  div:nth-child(4) {
    transform: rotate(90deg);

    &::before {
      animation-delay: -0.9s;
    }
  }

  div:nth-child(5) {
    transform: rotate(120deg);

    &::before {
      animation-delay: -0.8s;
    }
  }

  div:nth-child(6) {
    transform: rotate(150deg);

    &::before {
      animation-delay: -0.7s;
    }
  }

  div:nth-child(7) {
    transform: rotate(180deg);

    &::before {
      animation-delay: -0.6s;
    }
  }

  div:nth-child(8) {
    transform: rotate(210deg);

    &::before {
      animation-delay: -0.5s;
    }
  }

  div:nth-child(9) {
    transform: rotate(240deg);

    &::before {
      animation-delay: -0.4s;
    }
  }

  div:nth-child(10) {
    transform: rotate(270deg);

    &::before {
      animation-delay: -0.3s;
    }
  }

  div:nth-child(11) {
    transform: rotate(300deg);

    &::before {
      animation-delay: -0.2s;
    }
  }

  div:nth-child(12) {
    transform: rotate(330deg);

    &::before {
      animation-delay: -0.1s;
    }
  }
  @keyframes sk-circleFadeDelay {
    0%,
    39%,
    100% {
      opacity: 0;
    }

    40% {
      opacity: 1;
    }
  }
`
export const CircleSpinner = () => (
  <CircleSpinnerContainer>
    <div />
    <div />
    <div />
    <div />
    <div />
    <div />
    <div />
    <div />
    <div />
    <div />
    <div />
    <div />
  </CircleSpinnerContainer>
)
